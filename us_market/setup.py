"""Create private US DB credentials and start ONLY the dedicated Compose project."""
import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess

from .config import REPO


def setup(directory):
    directory=directory.expanduser().absolute()
    if directory.resolve().is_relative_to(REPO) or any(p.is_symlink() for p in [directory,*directory.parents]):
        raise ValueError("US DB secrets must live outside the repository, without symbolic links")
    if directory.exists():
        raise ValueError("Credential directory already exists; use the documented restart command, never replace secrets")
    # Avoid attaching an unrelated/orphaned database volume to new credentials.
    found=subprocess.run(["docker","volume","inspect","stock-bot-us-v1_us_mysql_data"],capture_output=True)
    if found.returncode==0:
        raise ValueError("US Docker volume already exists; inspect it before provisioning credentials")
    subprocess.run(["docker","info","--format","{{.ServerVersion}}"],check=True,capture_output=True)
    directory.mkdir(parents=True,mode=0o700)
    user_password,root_password=secrets.token_urlsafe(36),secrets.token_urlsafe(36)
    values={"user-password":user_password,"root-password":root_password,
            "db.json":json.dumps(dict(host="127.0.0.1",port=3308,database="stock_bot_us",user="us_market",password=user_password))}
    for name,value in values.items():
        fd=os.open(directory/name,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        with os.fdopen(fd,"w") as f: f.write(value+"\n")
    env=dict(os.environ,US_MYSQL_PASSWORD_FILE=str(directory/"user-password"),
             US_MYSQL_ROOT_PASSWORD_FILE=str(directory/"root-password"))
    subprocess.run(["docker","compose","-f",str(REPO/"us_market/compose.yaml"),"up","-d","--pull","never"],env=env,check=True)
    print(json.dumps(dict(project="stock-bot-us-v1",port=3308,config_file=str(directory/"db.json"),taiwan_modified=False)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,default=Path.home()/".config/stock_bot/us-market-v1")
    args=p.parse_args()
    try: setup(args.directory)
    except Exception as exc:
        print(json.dumps(dict(error_type=type(exc).__name__,error="US DB provisioning stopped; existing credentials/data were not replaced")))
        return 1
    return 0


if __name__=="__main__": raise SystemExit(main())
