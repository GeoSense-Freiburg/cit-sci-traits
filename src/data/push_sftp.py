from pathlib import Path

from src.conf.conf import get_config
from src.conf.environment import log
from src.io.upload_sftp import upload_file_sftp
from src.utils.dataset_utils import get_final_fns

if __name__ == "__main__":
    log.info("Pushing files to SFTP server")
    cfg = get_config()

    for fn in get_final_fns():
        log.info("Uploading %s...", fn)
        upload_file_sftp(
            fn,
            str(
                Path(
                    cfg.public.sftp_dir,
                    cfg.PFT,
                    f"{cfg.model_res}deg",
                    fn.name,
                )
            ),
        )
