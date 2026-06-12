# AWS SSH Setup

The AWS instance is Ubuntu 24.04 and will be accessed over SSH.

## Required Local Setup

Create or update `~/.ssh/config` on your local machine:

```sshconfig
Host isaac-aws
  HostName EC2_PUBLIC_IP_OR_DNS
  User ubuntu
  IdentityFile ~/.ssh/YOUR_KEY.pem
  IdentitiesOnly yes
  ServerAliveInterval 30
```

Use any alias name you prefer. Do not put the private key in this repository.

Recommended AWS security-group rule:

- Inbound TCP `22`
- Source: your current public IP only, not `0.0.0.0/0`

## Verify Connection

From a terminal on this machine:

```bash
ssh isaac-aws 'hostname && uname -a && nvidia-smi'
```

If that works, give me only the alias name, for example:

```text
isaac-aws
```

Then I can run non-destructive probes through that alias from this workspace, request approval for any command that needs network or remote access, and sync the repo to the server.

## Remote Probe

After the repo is synced to AWS, run:

```bash
bash scripts/probe_remote.sh
```

This checks:

- OS version
- GPU visibility
- NVIDIA driver
- Docker availability
- Isaac Sim launchers
- disk space

## Remote Sync

After SSH works:

```bash
bash scripts/sync_to_remote.sh isaac-aws ~/Isacc_dataset
```

The sync script excludes generated datasets and local environment files.
