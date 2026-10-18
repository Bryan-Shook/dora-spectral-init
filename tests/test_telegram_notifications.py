import importlib.util
import io
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

SPEC = importlib.util.spec_from_file_location(
    "study_notifications", Path(__file__).parents[1] / "scripts/run_study.py"
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)

TOKEN = "123456:test_token"
SETTINGS = (TOKEN, "12345")


@pytest.fixture
def credentials(tmp_path, monkeypatch):
    path = tmp_path / "telegram.local.json"
    monkeypatch.setattr(launcher, "TELEGRAM_CONFIG", path)
    return path


def test_missing_credentials_disable_alerts_without_network_access(credentials, monkeypatch):
    monkeypatch.setattr(launcher, "urlopen", lambda *a, **kw: pytest.fail("unexpected HTTP"))
    assert launcher.telegram_settings() is None
    monkeypatch.setattr(sys, "argv", ["run_study.py", "screen"])
    monkeypatch.setattr(launcher.study, "main", lambda: 0)
    assert launcher.main() == 0
    credentials.write_text(json.dumps({"bot_token": TOKEN, "chat_id": -12345}))
    assert launcher.telegram_settings() == (TOKEN, -12345)


@pytest.mark.parametrize(
    "content",
    [
        '{"bot_token":"123456:test_token",',
        json.dumps({"bot_token": TOKEN}),
        json.dumps({"bot_token": [TOKEN], "chat_id": 12345}),
        json.dumps({"bot_token": TOKEN, "chat_id": True}),
        json.dumps({"bot_token": TOKEN, "chat_id": ""}),
    ],
)
def test_invalid_config_is_nonfatal_and_does_not_expose_credentials(credentials, content, capsys):
    credentials.write_text(content)
    assert launcher.telegram_settings() is None
    warning = capsys.readouterr().err
    assert "alerts disabled" in warning
    assert TOKEN not in warning


def test_notification_uses_https_post_and_bounded_timeout(monkeypatch, capsys):
    calls = []

    def request(req, timeout):
        calls.append((req, timeout))
        return io.BytesIO(b'{"ok":true,"result":{"message_id":1}}')

    monkeypatch.setattr(launcher, "urlopen", request)
    assert launcher.send_telegram(SETTINGS, "screen completed")
    req, timeout = calls[0]
    assert req.full_url == f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    assert req.get_method() == "POST"
    assert json.loads(req.data) == {"chat_id": "12345", "text": "screen completed"}
    assert req.headers["Content-type"] == "application/json"
    assert timeout == 10
    assert TOKEN not in capsys.readouterr().out


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError(TOKEN),
        URLError(TOKEN),
        HTTPError(f"https://api.telegram.org/bot{TOKEN}/sendMessage", 403, TOKEN, {}, None),
    ],
)
def test_notification_failures_are_nonfatal_and_redact_token(monkeypatch, capsys, error):
    def request(*args, **kwargs):
        raise error

    monkeypatch.setattr(launcher, "urlopen", request)
    assert launcher.send_telegram(SETTINGS, "screen stopped") is False
    warning = capsys.readouterr().err
    assert "study result unchanged" in warning
    assert TOKEN not in warning


@pytest.mark.parametrize("response", [b'{"ok":false}', b"[]", b"not json"])
def test_api_rejection_is_not_reported_as_success(monkeypatch, response, capsys):
    monkeypatch.setattr(launcher, "urlopen", lambda *a, **kw: io.BytesIO(response))
    assert launcher.send_telegram(SETTINGS, "screen completed") is False
    assert "Alert sent" not in capsys.readouterr().out


@pytest.mark.parametrize("result", [0, 7])
def test_stage_sends_one_alert_after_all_work_and_preserves_exit_code(monkeypatch, result):
    monkeypatch.setattr(sys, "argv", ["run_study.py", "--directory", "runs/custom", "screen"])
    monkeypatch.setattr(launcher, "telegram_settings", lambda: SETTINGS)
    monkeypatch.setattr(launcher.getpass, "getuser", lambda: "researcher")
    monkeypatch.setattr(launcher.socket, "gethostname", lambda: "gpu-host")
    events = []

    def original_launch(*args):
        events.append("worker")
        return 0

    def monitor(command, output, original):
        return original(command, output)

    def study_main():
        launcher.study.launch([], Path("first.log"))
        launcher.study.launch([], Path("second.log"))
        events.append("evidence exported")
        return result

    def send(settings, message):
        assert settings == SETTINGS
        assert launcher.study.launch is original_launch
        assert events == ["worker", "worker", "evidence exported"]
        events.append(message)
        return False  # Delivery failure must not replace the study result.

    monkeypatch.setattr(launcher.study, "launch", original_launch)
    monkeypatch.setattr(launcher, "launch_with_progress", monitor)
    monkeypatch.setattr(launcher.study, "main", study_main)
    monkeypatch.setattr(launcher, "send_telegram", send)
    assert launcher.main() == result
    assert len(events) == 4
    assert ("completed" if result == 0 else "stopped") in events[-1]
    assert "Stage: screen" in events[-1] and "Study: runs/custom" in events[-1]
    assert "researcher@gpu-host" in events[-1] and "Elapsed:" in events[-1]
    assert launcher.study.launch is original_launch


@pytest.mark.parametrize(
    "error", [RuntimeError("allocation time limit reached"), KeyboardInterrupt()]
)
def test_stopped_stage_alert_preserves_original_exception(monkeypatch, error):
    monkeypatch.setattr(sys, "argv", ["run_study.py", "promote"])
    monkeypatch.setattr(launcher, "telegram_settings", lambda: SETTINGS)
    original = launcher.study.launch
    messages = []

    def study_main():
        raise error

    monkeypatch.setattr(launcher.study, "main", study_main)
    monkeypatch.setattr(
        launcher, "send_telegram", lambda settings, message: messages.append(message)
    )
    with pytest.raises(type(error)) as caught:
        launcher.main()
    assert caught.value is error
    assert launcher.study.launch is original
    assert len(messages) == 1
    assert "stopped" in messages[0] and type(error).__name__ in messages[0]
    assert "Stage: promote" in messages[0]


def test_unexpected_notification_error_cannot_override_success(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_study.py", "freeze"])
    monkeypatch.setattr(launcher, "telegram_settings", lambda: SETTINGS)
    monkeypatch.setattr(launcher.study, "main", lambda: 0)

    def broken(*args):
        raise OSError(TOKEN)

    monkeypatch.setattr(launcher, "send_telegram", broken)
    assert launcher.main() == 0


@pytest.mark.parametrize("arguments", [["status"], ["--help"], ["screen", "--help"]])
def test_help_and_status_do_not_load_credentials_or_send_alerts(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["run_study.py", *arguments])
    monkeypatch.setattr(launcher.study, "main", lambda: 0)
    monkeypatch.setattr(
        launcher, "telegram_settings", lambda: pytest.fail("credentials should not be loaded")
    )
    assert launcher.main() == 0


@pytest.mark.parametrize("delivered", [True, False])
def test_test_alert_does_not_start_experiments(monkeypatch, delivered):
    monkeypatch.setattr(sys, "argv", ["run_study.py", "--telegram-test"])
    monkeypatch.setattr(launcher, "telegram_settings", lambda: SETTINGS)
    monkeypatch.setattr(launcher.study, "main", lambda: pytest.fail("unexpected experiment"))
    messages = []

    def send(settings, message):
        messages.append(message)
        return delivered

    monkeypatch.setattr(launcher, "send_telegram", send)
    assert launcher.main() == (0 if delivered else 1)
    assert messages == ["DoRA study: Telegram test alert."]


def test_test_alert_requires_credentials_without_starting_experiments(monkeypatch, credentials):
    monkeypatch.setattr(sys, "argv", ["run_study.py", "--telegram-test"])
    monkeypatch.setattr(launcher.study, "main", lambda: pytest.fail("unexpected experiment"))
    monkeypatch.setattr(launcher, "urlopen", lambda *a, **kw: pytest.fail("unexpected HTTP"))
    assert launcher.main() == 1
