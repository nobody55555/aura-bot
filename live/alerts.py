"""
AlertManager – send critical notifications.

Primary channel: E-Mail (SMTP)
Credentials and server settings are read exclusively from environment
variables / .env – nothing sensitive is hardcoded or committed.

IRC is present only as a commented placeholder for future use.
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
from email.message import EmailMessage
from typing import Optional

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Minimal, reliable alerting.

    Required env vars for e-mail (all optional – if missing, alerts are logged only):
        ALERT_EMAIL_TO       recipient address
        ALERT_EMAIL_FROM     sender address
        SMTP_HOST            e.g. smtp.gmail.com / mail.infomaniak.com
        SMTP_PORT            usually 587 (STARTTLS) or 465 (SSL)
        SMTP_USER            login user (often same as FROM)
        SMTP_PASS            password or app-password
        SMTP_USE_TLS         "true" / "false" (default true for port 587)
    """

    def __init__(self):
        self.to_addr = os.getenv("ALERT_EMAIL_TO", "").strip()
        self.from_addr = os.getenv("ALERT_EMAIL_FROM", "").strip()
        self.smtp_host = os.getenv("SMTP_HOST", "").strip()
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "").strip()
        self.smtp_pass = os.getenv("SMTP_PASS", "")
        self.use_tls = os.getenv("SMTP_USE_TLS", "true").lower() in ("1", "true", "yes")

        self._email_enabled = bool(
            self.to_addr and self.from_addr and self.smtp_host and self.smtp_user
        )
        if not self._email_enabled:
            logger.info(
                "AlertManager: e-mail not fully configured "
                "(ALERT_EMAIL_TO / FROM / SMTP_HOST / SMTP_USER). "
                "Alerts will only be logged."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, subject: str, body: str, level: str = "INFO") -> bool:
        """
        Send an alert.
        Returns True if at least one channel succeeded.
        """
        full_subject = f"[Aura][{level}] {subject}"
        logger.log(
            logging.CRITICAL if level in ("CRITICAL", "HALT") else logging.WARNING,
            "ALERT %s: %s – %s", level, subject, body[:200],
        )

        ok = False
        if self._email_enabled:
            ok = self._send_email(full_subject, body) or ok

        # IRC placeholder – intentionally disabled / not implemented
        # if self._irc_enabled:
        #     ok = self._send_irc(full_subject, body) or ok

        return ok

    def critical(self, subject: str, body: str = "") -> bool:
        return self.send(subject, body or subject, level="CRITICAL")

    def halt(self, reason: str) -> bool:
        return self.critical("SYSTEM HALTED", reason)

    def warning(self, subject: str, body: str = "") -> bool:
        return self.send(subject, body or subject, level="WARNING")

    # ------------------------------------------------------------------
    # E-Mail implementation
    # ------------------------------------------------------------------

    def _send_email(self, subject: str, body: str) -> bool:
        if not self._email_enabled:
            return False
        try:
            msg = EmailMessage()
            msg["From"] = self.from_addr
            msg["To"] = self.to_addr
            msg["Subject"] = subject
            msg.set_content(body)

            if self.smtp_port == 465:
                # Implicit SSL
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, context=context, timeout=30) as server:
                    if self.smtp_pass:
                        server.login(self.smtp_user, self.smtp_pass)
                    server.send_message(msg)
            else:
                # STARTTLS (typical for 587)
                with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as server:
                    server.ehlo()
                    if self.use_tls:
                        context = ssl.create_default_context()
                        server.starttls(context=context)
                        server.ehlo()
                    if self.smtp_pass:
                        server.login(self.smtp_user, self.smtp_pass)
                    server.send_message(msg)

            logger.info("Alert e-mail sent to %s: %s", self.to_addr, subject)
            return True

        except Exception as e:
            logger.error("Failed to send alert e-mail: %s", e)
            return False

    # ------------------------------------------------------------------
    # IRC placeholder (not implemented – left for future)
    # ------------------------------------------------------------------

    # def _send_irc(self, subject: str, body: str) -> bool:
    #     """
    #     Optional IRC notification.
    #     Would read IRC_SERVER, IRC_PORT, IRC_CHANNEL, IRC_NICK, IRC_PASSWORD
    #     from environment. Left as placeholder only.
    #     """
    #     raise NotImplementedError("IRC alerting is not implemented")
