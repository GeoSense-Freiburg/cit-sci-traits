import os
import time
from pathlib import Path

import paramiko

from src.conf.environment import log


def upload_file_sftp(
    local_file_path: str | Path, remote_path: str, max_retries: int = 5
) -> None:
    """Upload a file to a remote server using SFTP."""
    sftp_server = os.environ["SFTP_SERVER"]
    port = int(os.environ["SFTP_PORT"])  # Default SFTP port
    username = os.environ["SFTP_USER"]
    ssh_key = paramiko.Ed25519Key.from_private_key_file(os.environ["SSH_KEY_PATH"])

    # Initialize SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    retries = 0
    while retries < max_retries:
        try:
            # Connect to the server
            ssh.connect(sftp_server, port=port, username=username, pkey=ssh_key)

            # Initialize SFTP session
            sftp = ssh.open_sftp()

            # Upload file
            sftp.put(local_file_path, remote_path)
            log.info("Successfully uploaded %s to %s", local_file_path, remote_path)

            # Close SFTP session and SSH connection
            sftp.close()
            ssh.close()
            return
        except paramiko.SSHException:
            retries += 1
            log.warning(
                "SSHException encountered. Retrying in 60 seconds... (Attempt %d/%d)",
                retries,
                max_retries,
            )
            time.sleep(60)
        except Exception as e:
            log.error("Failed to upload file: %s", e)
            ssh.close()
            return

    log.error("Failed to upload file after %d attempts", max_retries)
