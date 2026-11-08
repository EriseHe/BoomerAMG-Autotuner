# AWS migration from the selected commit

Base: `5b111d1f4978865f8b76a989f1b3637121ab7e11`.
This folder prepares a pilot; it does not provision, upload, authenticate, or
change account permissions. AWS account access, region, AMI, instance type,
network, storage destination and pilot budget still require selection/approval.
The Mac is not a controller for cloud jobs. Once an approved EC2 job boots, its
build, execution, result upload and shutdown run entirely on that instance.

## Scope and exact deviations

The 36 jobs are diffusion/advection × 40³/60³/80³ × seeds 1–6. Each original
configuration is referenced unchanged: 5,000 online inputs + 1,000 frozen inputs,
all four original methods, original seeds, candidate generation, features,
learning/recovery rules, tolerance and timing accounting. Methods remain serial
within each input, in the original balanced permutation order. No MPI scaling.

Only two existing files differ from the base:

1. `run_online.py` selects the platform's actual runtime library for its source
   snapshot symlink, and records the selected base commit from `DEPLOYMENT.json`
   when Git history is absent. Actual patched source hashes remain recorded by
   the original runner; the deployment manifest records both base/current hashes
   and the complete patch. This does not claim the patch is part of the base.
2. `hypre/bindings/Makefile` selects `.so` on Linux and `.dylib` on macOS for its
   compatibility build entry point. HYPRE and the numerical implementation are
   unchanged. The main `make -C hypre` entry already supports Linux.

All other additions are in `deployment/aws/`. `portability.patch` is generated
inside the bundle. No historical result/checkpoint is used to initialize a job.

## Local preparation (no native solves)

From this isolated repository root:

```sh
python3 deployment/aws/campaign.py scan
python3 deployment/aws/campaign.py plan
python3 -m unittest discover -s deployment/aws -p 'test_*.py' -v
python3 deployment/aws/campaign.py bundle --output /absolute/new/source.tgz
```

The scanner reads every tracked blob at the base commit and reports only file,
line and rule, never values. It also scans the outgoing files. Common AWS,
GitHub, Slack, Google key, URL-password and private-key patterns plus sensitive
filenames are covered. This is not a forensic guarantee or a history audit. The
bundle is allowlisted, deterministic, checksummed, and contains no `.git`, local
environment, secrets directories, old results, native binaries or historical
measurements. Only pinned input captures and their manifests are copied from
`data/`. Bundling fails on unexpected tracked source changes or findings.

`plan/campaign.json` and the 36 singleton suites are generated from the original
configs and carry their SHA-256 hashes. Do not modify a config for cloud use.

## Linux pilot sequence (future approved work)

Use a fresh Ubuntu 22.04 LTS EBS-backed AMI of the approved CPU architecture,
with systemd and SSM Agent already configured. Verify the AMI's architecture,
owner and root-device name using AWS before rendering a request. Linux has not
yet been tested here. Match the exact AMI/instance family/build environment for
all comparison jobs. Do not mix ARM and x86 results into one hardware cohort.

Bootstrap installs OS build tools and a private Python environment on the NEW
instance, builds pinned Python dependencies and this HYPRE source, then executes
as an ordinary non-root user. It installs nothing on the Mac. For the first
pilot choose smoke mode (24 inputs); smoke cannot test the 5,000→5,001 freeze
boundary or provide paper results. Run a complete 6,000-input singleton pilot
after smoke passes, and verify freeze/checkpoint/completion audits before scale.
Validate an 80³ pilot's RSS before applying a RAM assumption to all jobs.

Suggested planning inputs, not approved purchases: user-reported ~2 GB/job and
~1/2/6 hours at 40³/60³/80³. These are not Linux measurements. A 4–8 GiB pilot
could be evaluated against that report; final RAM/CPU choice remains pending.
Use a non-burstable CPU type for stable timing and avoid interruptible capacity
until restart support exists. No GPU is used. Default job limits are 4/6/12 hours
for full runs and 30 minutes for smoke; these are deadlines, not predictions.
The external systemd watchdog allows another 130 minutes for setup/uploads.

Build output includes the OS package versions, compiler command, CMake cache,
Python/MPI versions, source/native hashes and thread settings. OS package and
build-tool versions resolve at pilot time, so bootstrap is not yet a fully
locked machine image. Preserve that evidence and use one approved prebuilt AMI
for the full campaign if strict compiler/OS identity is required.

After explicit approval, upload the checked bundle to the approved S3 prefix.
Use a new campaign ID for each repeat; no job output is reused. Render a SINGLE
pilot request (the renderer makes no AWS calls):

```sh
python3 deployment/aws/campaign.py launch-request \
  --campaign APPROVED_UNIQUE_ID --job diffusion_40_seed1 --mode smoke \
  --ami APPROVED_AMI --instance-type APPROVED_TYPE --region APPROVED_REGION \
  --subnet APPROVED_SUBNET --security-group APPROVED_SG \
  --instance-profile APPROVED_INSTANCE_PROFILE --root-device /dev/sda1 \
  --bundle-uri s3://APPROVED_BUCKET/source/source.tgz \
  --bundle-sha256 THE_64_CHARACTER_CHECKSUM \
  --output-uri s3://APPROVED_BUCKET/results --output /absolute/new/pilot-request
```

Replace placeholders; the bucket must be lowercase and the hash valid. Public
IPv4 is off unless `--public-ip` is explicitly selected. The existing subnet
must provide approved egress to S3, Ubuntu/PyPI and SSM (NAT/endpoints or public
networking as appropriate). The tooling creates no VPC/security-group rules or
SSH keys. IMDSv2 is required; use an instance profile, not copied credentials.

After reviewing the rendered JSON and user data, use an authenticated AWS
CloudShell/approved CLI session to check permission WITHOUT launching:

```sh
aws ec2 run-instances --region APPROVED_REGION \
  --cli-input-json file:///absolute/pilot-request/run-instances.json --dry-run
```

`DryRunOperation` means this permission check passed (the CLI returns a nonzero
status for that response). It does not validate bootstrap, budget, availability,
quotas or scientific correctness. Remove `--dry-run` ONLY after pilot spending
approval. Save the launch response and instance ID. Each request has count=1
and an idempotency token. No code here launches a 36-instance fleet.

## Result integrity, timeouts and cleanup

Source files must match `DEPLOYMENT.json` before any job. Each singleton uses a
fresh output directory, subprocess group and hard deadline. Timeout terminates
only that group's launcher/worker. GNU time records elapsed/CPU/max RSS for
prepare and execution; max RSS is a per-process high-water mark, not total
simultaneous job memory. No CPU affinity is claimed or silently imposed.

Success requires original suite/seed completion, trajectory and checkpoint
hashes, all four method audits and, for full runs, the freeze boundary and frozen
state audits. `OUTPUT_SHA256.json` covers regular result files. Native warmup,
candidate generation and postprocessing are retained in the original protocol.

The EXIT handler uploads `results.tgz` and its checksum, downloads the archive
again, and compares bytes before publishing `upload_receipt.json`. It stops the
instance even on build/run/upload failure. If upload fails, EBS preserves local
evidence. The source snapshot symlinks are excluded from output archives; keep
the source bundle beside the results. Build evidence is included on completion
and on failures after it is collected. The hard-stop timer bounds a hung job;
it cannot promise a final upload if the machine/OS fails.

**Stopping ends compute billing but EBS and other provisioned resources remain
billable.** Cleanup is a separate explicit action after retrieval verification:

1. Retrieve the archive, checksum and upload receipt. Verify checksum and
   `job_receipt.json.passed` (smoke success remains smoke, not official data).
2. Verify `Project=RLTune-SISC`, exact `Campaign`, `Job`, base commit and stopped
   state on the EXACT instance IDs in the saved launch responses. Never select
   by a broad account-wide filter or terminate unrelated instances.
3. Request authorization before `terminate-instances`. Root EBS has
   DeleteOnTermination=true, so termination discards its recovery copy. Do not
   terminate if output retrieval/hash validation is missing or uncertain.
4. Check termination and tagged EBS cleanup. Retain source/results in S3; no
   S3 deletion or automatic account-wide cleanup is included.

Partial learned-state resume is explicitly unsupported by the official driver,
despite checkpoints every 1,000 inputs. Preserve failed evidence, use a new
campaign/output directory and restart that seed from input one. Never delete
partial files to force a resume.

## AWS permissions and user actions

An administrator/user must provide or approve the account, region and budget;
existing bucket/prefix, AMI, subnet, SG, instance type and instance profile; and
a short-lived authenticated CloudShell/SSO session. No access keys need to be
shared with this task or stored in the bundle. Check EC2 vCPU quotas for the
eventual 36 instances; a pilot requests only one.

Controller permissions: `sts:GetCallerIdentity`, EC2 Describe APIs,
`ec2:RunInstances`, tag-on-create `ec2:CreateTags`, and `iam:PassRole` restricted
to the approved EC2 role. Stop/terminate permissions should be scoped to campaign
tags and used only when authorized. S3 bundle upload/result retrieval need
PutObject/GetObject on the exact prefixes; ListBucket only if listing is used.

Instance role: GetObject on the single source object; PutObject AND GetObject on
its result prefix (round-trip verification needs read access). It needs no EC2
launch/terminate or IAM-management permission. Optional SSM uses the established
SSM instance policy and network connectivity; user session/Run Command access
is separate. SSE-KMS buckets require the corresponding narrowly scoped KMS
permissions; S3/EBS encryption policy must match the approved account setup.
The scripts do not create or widen any permissions.

Sources checked: [EC2 RunInstances](https://docs.aws.amazon.com/cli/latest/reference/ec2/run-instances.html),
[SSM instance permissions](https://docs.aws.amazon.com/systems-manager/latest/userguide/setup-instance-permissions.html),
[instance profiles](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_use_switch-role-ec2_instance-profiles.html),
[instance stop behavior](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Stop_Start.html).
