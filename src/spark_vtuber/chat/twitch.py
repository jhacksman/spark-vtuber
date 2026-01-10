"""
Twitch chat integration for Spark VTuber.

Implements Twitch IRC for chat interaction.
"""

import asyncio
import re
import uuid
from datetime import datetime
from typing import AsyncIterator

from spark_vtuber.chat.base import BaseChat, ChatMessage, MessageType


class TwitchChat(BaseChat):
    """
    Twitch chat client using IRC.

    Supports:
    - Reading chat messages
    - Sending messages
    - Parsing badges, emotes, and metadata
    - Handling subscriptions, raids, etc.
    """

    IRC_HOST = "irc.chat.twitch.tv"
    IRC_PORT = 6667
    IRC_PORT_SSL = 6697

    def __init__(
        self,
        channel: str,
        oauth_token: str,
        nickname: str = "sparkbot",
        use_ssl: bool = True,
        **kwargs,
    ):
        """
        Initialize Twitch chat client.

        Args:
            channel: Twitch channel name (without #)
            oauth_token: OAuth token (oauth:xxx format)
            nickname: Bot nickname
            use_ssl: Whether to use SSL connection
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.channel = channel.lower().lstrip("#")
        self.oauth_token = oauth_token
        self.nickname = nickname.lower()
        self.use_ssl = use_ssl
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._message_queue: asyncio.Queue[ChatMessage] = asyncio.Queue()
        self._running = False

    async def connect(self) -> None:
        """Connect to Twitch IRC."""
        if self._connected:
            self.logger.warning("Already connected to Twitch")
            return

        self.logger.info(f"Connecting to Twitch channel: {self.channel}")

        try:
            port = self.IRC_PORT_SSL if self.use_ssl else self.IRC_PORT

            if self.use_ssl:
                import ssl
                ssl_context = ssl.create_default_context()
                self._reader, self._writer = await asyncio.open_connection(
                    self.IRC_HOST,
                    port,
                    ssl=ssl_context,
                )
            else:
                self._reader, self._writer = await asyncio.open_connection(
                    self.IRC_HOST,
                    port,
                )

            await self._authenticate()

            await self._request_capabilities()

            await self._join_channel()

            self._connected = True
            self._running = True

            asyncio.create_task(self._read_loop())

            self.logger.info(f"Connected to Twitch channel: {self.channel}")

        except Exception as e:
            self.logger.error(f"Failed to connect to Twitch: {e}")
            raise

    async def _authenticate(self) -> None:
        """Authenticate with Twitch IRC."""
        token = self.oauth_token
        if not token.startswith("oauth:"):
            token = f"oauth:{token}"

        self._writer.write(f"PASS {token}\r\n".encode())
        self._writer.write(f"NICK {self.nickname}\r\n".encode())
        await self._writer.drain()

    async def _request_capabilities(self) -> None:
        """Request Twitch IRC capabilities."""
        caps = [
            "twitch.tv/membership",
            "twitch.tv/tags",
            "twitch.tv/commands",
        ]
        for cap in caps:
            self._writer.write(f"CAP REQ :{cap}\r\n".encode())
        await self._writer.drain()

    async def _join_channel(self) -> None:
        """Join the Twitch channel."""
        self._writer.write(f"JOIN #{self.channel}\r\n".encode())
        await self._writer.drain()

    async def disconnect(self) -> None:
        """Disconnect from Twitch IRC."""
        if not self._connected:
            return

        self.logger.info("Disconnecting from Twitch")

        self._running = False

        if self._writer:
            self._writer.write(f"PART #{self.channel}\r\n".encode())
            await self._writer.drain()
            self._writer.close()
            await self._writer.wait_closed()

        self._reader = None
        self._writer = None
        self._connected = False

    async def send_message(self, content: str) -> None:
        """Send a message to chat."""
        if not self._connected or not self._writer:
            raise RuntimeError("Not connected to Twitch")

        self._writer.write(f"PRIVMSG #{self.channel} :{content}\r\n".encode())
        await self._writer.drain()

    async def get_messages(self) -> AsyncIterator[ChatMessage]:
        """Get incoming messages."""
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0,
                )
                yield message
            except asyncio.TimeoutError:
                continue

    async def _read_loop(self) -> None:
        """Read messages from IRC."""
        while self._running and self._reader:
            try:
                line = await self._reader.readline()
                if not line:
                    break

                line = line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                if line.startswith("PING"):
                    pong = line.replace("PING", "PONG")
                    self._writer.write(f"{pong}\r\n".encode())
                    await self._writer.drain()
                    continue

                message = self._parse_message(line)
                if message:
                    await self._message_queue.put(message)
                    await self._dispatch_message(message)

            except Exception as e:
                self.logger.error(f"Error reading from Twitch: {e}")
                if not self._running:
                    break

    def _parse_message(self, line: str) -> ChatMessage | None:
        """Parse an IRC message line."""
        tags = {}
        if line.startswith("@"):
            tag_end = line.index(" ")
            tag_str = line[1:tag_end]
            line = line[tag_end + 1:]

            for tag in tag_str.split(";"):
                if "=" in tag:
                    key, value = tag.split("=", 1)
                    tags[key] = value

        privmsg_match = re.match(
            r":(\w+)!\w+@\w+\.tmi\.twitch\.tv PRIVMSG #(\w+) :(.+)",
            line,
        )

        if not privmsg_match:
            return None

        username = privmsg_match.group(1)
        channel = privmsg_match.group(2)
        content = privmsg_match.group(3)

        display_name = tags.get("display-name", username)
        message_id = tags.get("id", str(uuid.uuid4()))

        badges = []
        if "badges" in tags and tags["badges"]:
            badges = [b.split("/")[0] for b in tags["badges"].split(",")]

        emotes = {}
        if "emotes" in tags and tags["emotes"]:
            for emote_data in tags["emotes"].split("/"):
                if ":" in emote_data:
                    emote_id, positions = emote_data.split(":", 1)
                    emote_positions = []
                    for pos in positions.split(","):
                        if "-" in pos:
                            start, end = pos.split("-")
                            emote_positions.append((int(start), int(end)))
                    emotes[emote_id] = emote_positions

        message_type = MessageType.CHAT
        if content.startswith("!"):
            message_type = MessageType.COMMAND

        return ChatMessage(
            id=message_id,
            platform="twitch",
            username=username,
            display_name=display_name,
            content=content,
            timestamp=datetime.now(),
            message_type=message_type,
            is_moderator="moderator" in badges or "broadcaster" in badges,
            is_subscriber="subscriber" in badges,
            is_vip="vip" in badges,
            badges=badges,
            emotes=emotes,
            metadata={"tags": tags},
        )
