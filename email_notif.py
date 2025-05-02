import subprocess
import smtplib
import time
from email.message import EmailMessage
import socket


CHECK_INTERVAL = 60  # seconds
EMAIL_SENT = False  # track whether we already sent notification

GMAIL_USER = "filip.masny@gmail.com"
GMAIL_PASSWORD = "llvk qfph tcgp hjrq"

hostname = socket.gethostname()


def get_idle_gpus():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        active_gpus = set(result.stdout.strip().splitlines())

        gpu_info = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        all_gpus = set(gpu_info.stdout.strip().splitlines())

        idle_gpus = all_gpus - active_gpus
        return idle_gpus
    except Exception as e:
        print(f"❌ Error checking GPUs: {e}")
        return set()


def send_email(subject, body, to_emails):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = ", ".join(to_emails)
    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(GMAIL_USER, GMAIL_PASSWORD)
            smtp.send_message(msg)
        print("✅ Email sent.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")


# --- Real-time loop ---
print("🔄 Starting GPU idle checker...")
while True:
    idle = get_idle_gpus()
    if idle and not EMAIL_SENT:
        recipients = [GMAIL_USER, "masny5@uniba.sk"]
        print(f"✅ Idle GPU(s) found: {idle}")
        send_email(
            subject=f"🚀 GPU Available on {hostname}!",
            body=f"The following GPU(s) are idle on {hostname}:\n" + "\n".join(idle),
            to_emails=[GMAIL_USER],
        )
        EMAIL_SENT = True
    elif not idle:
        print("🟡 All GPUs busy.")
        EMAIL_SENT = False  # reset so we can notify again when they free up

    time.sleep(CHECK_INTERVAL)
