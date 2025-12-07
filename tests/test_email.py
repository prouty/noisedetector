"""
Tests for core.email module.

Tests email configuration and sending functionality.
"""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from core.email import get_email_config, send_email


class TestGetEmailConfig:
    """Test email configuration loading."""
    
    def test_get_email_config_from_config(self, config):
        """Test loading email config from config.json."""
        if not config or "email" not in config:
            pytest.skip("No email config in config.json")
        
        email_config = get_email_config(config)
        
        assert isinstance(email_config, dict)
        assert "smtp_server" in email_config
        assert "smtp_port" in email_config
        assert "to_address" in email_config
    
    def test_get_email_config_from_env(self):
        """Test loading email config from environment variables."""
        # Set environment variables
        env_vars = {
            "EMAIL_SMTP_SERVER": "smtp.test.com",
            "EMAIL_SMTP_PORT": "465",
            "EMAIL_SMTP_USERNAME": "testuser",
            "EMAIL_SMTP_PASSWORD": "testpass",
            "EMAIL_FROM": "from@test.com",
            "EMAIL_TO": "to@test.com",
            "EMAIL_USE_TLS": "false",
        }
        
        with patch.dict(os.environ, env_vars):
            email_config = get_email_config()
            
            assert email_config["smtp_server"] == "smtp.test.com"
            assert email_config["smtp_port"] == 465
            assert email_config["smtp_username"] == "testuser"
            assert email_config["smtp_password"] == "testpass"
            assert email_config["from_address"] == "from@test.com"
            assert email_config["to_address"] == "to@test.com"
            assert email_config["use_tls"] is False
    
    def test_get_email_config_env_overrides_config(self):
        """Test that environment variables override config file."""
        config = {
            "email": {
                "smtp_server": "config.server.com",
                "smtp_port": 587,
                "to_address": "config@test.com",
            }
        }
        
        env_vars = {
            "EMAIL_SMTP_SERVER": "env.server.com",
            "EMAIL_TO": "env@test.com",
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            email_config = get_email_config(config)
            
            # Environment should override
            assert email_config["smtp_server"] == "env.server.com"
            assert email_config["to_address"] == "env@test.com"
            # Config value should remain for non-overridden keys
            assert email_config["smtp_port"] == 587
    
    def test_get_email_config_defaults(self):
        """Test default values when no config provided."""
        with patch.dict(os.environ, {}, clear=True):
            email_config = get_email_config()
            
            assert email_config["smtp_port"] == 587  # Default port
            assert email_config["use_tls"] is True  # Default TLS
            assert email_config.get("smtp_server") == ""  # Empty if not set


class TestSendEmail:
    """Test email sending functionality."""
    
    @patch('core.email.smtplib.SMTP')
    def test_send_email_success(self, mock_smtp_class):
        """Test successful email sending."""
        # Setup mock SMTP server
        mock_server = MagicMock()
        mock_smtp_class.return_value = mock_server
        
        email_config = {
            "smtp_server": "smtp.test.com",
            "smtp_port": 587,
            "smtp_username": "user",
            "smtp_password": "pass",
            "from_address": "from@test.com",
            "to_address": "to@test.com",
            "use_tls": True,
        }
        
        report_text = "Test report content"
        result = send_email(report_text, email_config)
        
        assert result is True
        mock_smtp_class.assert_called_once_with("smtp.test.com", 587)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("user", "pass")
        mock_server.send_message.assert_called_once()
        mock_server.quit.assert_called_once()
    
    @patch('core.email.smtplib.SMTP')
    def test_send_email_no_tls(self, mock_smtp_class):
        """Test email sending without TLS."""
        mock_server = MagicMock()
        mock_smtp_class.return_value = mock_server
        
        email_config = {
            "smtp_server": "smtp.test.com",
            "smtp_port": 465,
            "smtp_username": "user",
            "smtp_password": "pass",
            "from_address": "from@test.com",
            "to_address": "to@test.com",
            "use_tls": False,
        }
        
        result = send_email("Test", email_config)
        
        assert result is True
        # Should not call starttls when use_tls is False
        mock_server.starttls.assert_not_called()
        mock_server.login.assert_called_once_with("user", "pass")
    
    @patch('core.email.smtplib.SMTP')
    def test_send_email_missing_config(self, mock_smtp_class):
        """Test email sending with missing required config."""
        email_config = {
            "smtp_server": "",  # Missing server
            "to_address": "",   # Missing recipient
        }
        
        result = send_email("Test", email_config)
        
        assert result is False
        mock_smtp_class.assert_not_called()
    
    @patch('core.email.smtplib.SMTP')
    def test_send_email_smtp_error(self, mock_smtp_class):
        """Test handling of SMTP login errors."""
        mock_server = MagicMock()
        # Login error should cause function to raise and return False
        # Use smtplib.SMTPAuthenticationError to be more realistic
        from smtplib import SMTPAuthenticationError
        mock_server.login.side_effect = SMTPAuthenticationError(535, "Authentication failed")
        mock_smtp_class.return_value = mock_server
        
        email_config = {
            "smtp_server": "smtp.test.com",
            "smtp_port": 587,
            "smtp_username": "user",
            "smtp_password": "pass",
            "from_address": "from@test.com",
            "to_address": "to@test.com",
            "use_tls": True,
        }
        
        result = send_email("Test", email_config)
        
        assert result is False
        mock_server.login.assert_called_once()
        # Should not have called send_message if login failed
        mock_server.send_message.assert_not_called()
    
    @patch('core.email.smtplib.SMTP')
    def test_send_email_cleanup_on_error(self, mock_smtp_class):
        """Test that cleanup happens even on send error."""
        mock_server = MagicMock()
        # Make send_message raise a non-535 error (real send failure, not FastMail quirk)
        # Use SMTPException to be more realistic
        from smtplib import SMTPException
        mock_server.send_message.side_effect = SMTPException(550, "Mailbox unavailable")
        mock_smtp_class.return_value = mock_server
        
        email_config = {
            "smtp_server": "smtp.test.com",
            "smtp_port": 587,
            "smtp_username": "user",
            "smtp_password": "pass",
            "from_address": "from@test.com",
            "to_address": "to@test.com",
            "use_tls": True,
        }
        
        result = send_email("Test", email_config)
        
        assert result is False
        # Should have attempted to send
        mock_server.send_message.assert_called_once()
        # Should still attempt cleanup
        mock_server.quit.assert_called_once()

