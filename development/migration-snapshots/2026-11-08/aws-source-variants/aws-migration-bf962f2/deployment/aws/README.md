# AWS migration from the selected commit

Base: `bf962f2d990f22d2abc94043db0ef4a84aa277fe`.
This supersedes the earlier preparation from `5b111d1`. The prior clone/bundle
is preserved for audit, but must not be used for the updated campaign. The new
base is exactly one commit above it; only the two upstream READMEs and new
`experiments/run_all.py` changed. The new wrapper itself is included unchanged.
This folder prepares a pilot; it does not provision, upload, authenticate, or
change account permissions. The user approved up to $10 total usage for the
six-job serial correctness pilot below. Actual AMI, instance/rates, networking,
storage destination and bounded cleanup plan still require validation. The
broader 36-job campaign has no spending approval.
The Mac is not a controller for cloud jobs. Once an approved EC2 job boots, its
build, execution, result upload and shutdown run entirely on that instance.

Deployment revision r3 adds an immediate shutdown fallback after the Linux/root
precondition and arms the independent timer before required-setting checks,
directory creation or logging setup. Failure to register the timer requests an
immediate stop. Its optional timeout marker has a two-second deadline and a
one-second kill grace; it cannot gate the subsequent absolute shutdown command.
There is no explicit unbounded sync before the backup stop. The cleanup
attestation guard, six-job order, official configs and scientific source are
unchanged. Prior r2 archives and proposals remain preserved.

## Approved serial pilot: six full jobs on one VM

Use `serial_pilot.py`, not `run_all.py`, three-seed suites, or the generic
singleton request example, for the newly approved pilot. Order is exactly:
diffusion40 seed1, advection40 seed1, diffusion40 seed2, advection40 seed2,
diffusion40 seed3, advection40 seed3. `serial-plan.json` records this scope.
Each is a fresh full official 5,000-learning + 1,000-frozen stream and retains
all four methods. No official config/math is changed. One seed process runs
at a time, and all numerical-library thread limits stay at one.

The seed-1 pair runs first. Only complete correctness success (all case counts,
four method/checkpoint/trajectory/freeze audits and zero final failures) allows
seeds 2/3. Any failure aborts the remaining queue and preserves existing output.
A host-wide pilot lock prevents overlapping serial-pilot invocations. Flex
measurements are explicitly exploratory correctness/memory evidence, not
publishable timing comparisons. Two-second descendant RSS samples supplement
GNU-time per-process high-water marks; summed RSS double-counts shared pages
and sampling may miss short peaks. Outputs and receipts retain both caveats.

`pilot-budget.template.json` is deliberately unvalidated and **cannot generate
a request**. After the parent/browser task provides the exact resource plan,
copy it to a separate budget file and enter checked Ohio EC2, EBS and IPv4 rates,
public-IP/disk choices, source URLs/time, a reserve for storage/requests/transfers
and any networking fees, and a confirmed <=24-hour post-stop cleanup plan.
Rates use gross usage before credits. The guard adds a billed-hour buffer and
retained-storage/IP costs, rejects a ceiling above $10, rejects multiple VMs,
non-Free-plan types, stale rates and account upgrades. A cleanup commitment is
an external requirement: the process watchdog alone cannot cap ongoing EBS/S3
charges after shutdown. Do not claim a hard all-cost guarantee until that
resource/cleanup plan is validated. No estimated hourly rate is embedded.

The parent account inspector has checked the proposed pilot host: Ohio
`us-east-2`, `m7i-flex.large` (2 vCPU/8 GiB), Ubuntu 24.04 x86 AMI
`ami-02ad619be5bc165df`, root `/dev/sda1`, and EC2 $0.09576/hour. Proposed
storage is encrypted 30 GiB gp3 (3,000 IOPS/125 MiB/s), with one public IPv4 at
$0.005/hour and no NAT. Exact budget/policy artifacts beside the current bundle
use these supplied facts. VPC/subnet are existing inspected resources; the S3
bucket, role/profile and security group remain proposals, not created resources.
The 24-hour cleanup assumption still needs an explicit commitment before the
cost guard may accept a final request. Source upload, IAM/network changes and
launch remain subject to the parent's per-action approval workflow.

Provisional whole-VM wall limit is eight hours, including bootstrap and upload;
the execution queue reserves the final 30 minutes for upload. Each job also
receives only the remaining execution window. A systemd hard-stop timer bounds
the VM lifetime independently of the Mac. Time limits may stop the pilot before
all six jobs finish; exceeding the allowance or restarting partial adaptive
state is not permitted. A failed upload retains EBS for recovery under the
validated cleanup deadline. Successful output is uploaded and read back for
byte comparison; the VM stops rather than silently terminating its recovery
copy. Root disk and S3 retention remain billable until their approved cleanup.
The systemd timer is owned by the VM's service manager, independent of an SSH
shell or Mac connection. At its hard deadline it attempts `HARD_TIMEOUT.txt`
within a bounded best-effort step and requests guest shutdown, allowing normal
OS filesystem shutdown. S3 upload may be incomplete at that
deadline; EBS is retained and must be recovered before termination. A timeout
after five successes cannot mark a six-job pilot as passed.

When resource values have actually been checked:

```sh
python3 deployment/aws/serial_pilot.py validate-budget --budget-file /approved/pilot-budget.json
python3 deployment/aws/serial_pilot.py request \
  --budget-file /approved/pilot-budget.json --campaign APPROVED_UNIQUE_ID \
  --ami APPROVED_AMI --instance-type APPROVED_ELIGIBLE_TYPE --region us-east-2 \
  --subnet APPROVED_SUBNET --security-group APPROVED_SG \
  --instance-profile APPROVED_INSTANCE_PROFILE --root-device APPROVED_ROOT_DEVICE \
  --bundle-uri s3://APPROVED_BUCKET/source/serial-pilot.tgz \
  --bundle-sha256 THE_64_CHARACTER_CHECKSUM --output-uri s3://APPROVED_BUCKET/results \
  --disk-gib 30 --output /absolute/new/serial-request
```

Add `--public-ip` only if the validated budget/network plan specifies it. This
renders ONE instance request; it launches nothing. The budget is embedded in
user data without credentials. Bootstrap invokes the serial runner on the VM;
it builds once, runs each singleton sequentially, archives all completed/failed
job directories and memory evidence, uploads, then stops. CloudShell/browser
launch and cleanup belong to the parent task; this preparation makes no AWS
calls and creates no credentials. Host facts are now supplied as above; resource
creation/launch approval and the cleanup commitment remain pending.

## Scope and exact deviations

The 36 jobs are diffusion/advection × 40³/60³/80³ × seeds 1–6. Each original
configuration is referenced unchanged: 5,000 online inputs + 1,000 frozen inputs,
all four original methods, original seeds, candidate generation, features,
learning/recovery rules, tolerance and timing accounting. Methods remain serial
within each input, in the original balanced permutation order. No MPI scaling.

**Those 36 jobs cover Module 1 only.** `experiments/run_all.py` is a useful
single-machine wrapper, not the per-instance entry point. Unchanged, it runs all
12 Module 1 three-seed suites, then Module 2 using all six diffusion-60 joint
outputs by default, then two Module 3 variable-flow three-seed suites. Running
it on each of 36 machines would repeat that entire study 36 times. AWS instances
continue to invoke the underlying `run_online prepare/run` with ONE seed suite.
`plan/full-study.json` records the extra dependencies and work without adding
anything to the executable 36-job campaign or approving any spending.

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

## Full-study dependency and collection plan

1. **Module 1:** 36 independent full jobs, each 5,000 learning + 1,000 frozen
   inputs; 864,000 method evaluations before retries. Every comparison method
   stays inside its singleton job. No extra Module 2/3 process is started by
   `campaign.py run` or the EC2 bootstrap.
2. **Module 2:** wait for the six FULL `diffusion_60_seed1` … `seed6` jobs to
   finish and pass frozen/completion/checkpoint audits. Smoke/partial runs,
   historical Mac checkpoints and other source revisions are rejected. Other
   Module 1 family/grids are not prerequisites; they may finish independently.
   Download each verified archive on a separate evaluation host and extract it
   under `COLLECTION_ROOT/diffusion_60_seedN/`. There is no dependency on a Mac
   filesystem path, and no need to recreate the old batch1/batch2 directory tree.

   ```sh
   python deployment/aws/campaign.py module2-collect \
     --inputs-root /absolute/collected --output /absolute/new/module2-plan
   ```

   This validates output inventories, full-run receipts, original configurations,
   trajectory/checkpoint hashes, freeze audits, and matching training source and
   native builds across all six sources. It writes `collection-plan.json` with
   explicit commands and absolute source paths; it runs no policies and makes no
   AWS calls. Keep extracted result directories in place after collection.
   Required files per source are the experiment configuration, final joint
   controller `.npz` plus its `.encoder.json`, and final joint setup-model `.npz`.
   Keep the full archive/audits for provenance, not just these four files.

   Review the emitted `run_policy init`, `validate`, `prepare`, and `run`
   commands and approve this additional evaluation before executing them.
   They preserve upstream random seed 20261019, 100 evaluation problems, all six
   checkpoints, 41 fixed weights, the 10% extra-repeat scan, and three evaluation
   repetitions. The scan has 29,520 policy trials. Evaluation has up to 9,000
   unique cells before recovery/preflight; identical policy cells are shared.
   Default execution uses three single-rank workers on one separate evaluation
   host. It is not another 36-machine campaign. Validate available CPU/RAM and
   record the evaluation environment; no Linux runtime or dollar cost is known.
   Module 2 supports its own resumable scan/test records, which does not add
   partial adaptive-state resume to Modules 1 or 3. Do not run Module 2's default
   three workers on a singleton VM accidentally.
3. **Module 3:** six NEW `variable_flow_60` seed runs, 5,000 learning inputs
   each, four cold methods and no frozen phase: 30,000 paired inputs / 120,000
   method evaluations before retries. These runs do not load Module 1/2 models
   and can run independently on separately approved resources. Their unchanged
   config hashes appear in `full-study.json`; they are deliberately NOT accepted
   job IDs by the current Module 1 AWS launcher. Singleton Module 3 suites and
   their launch/time/memory validation remain future work after scope approval.
   Runtime and cost are additional, unmeasured, and excluded from the 36-job
   Module 1 estimate.

Complete study therefore means **42 adaptive seed runs plus one separately
orchestrated matched-policy evaluation**, not 36 total runs or 43 machines that
must run simultaneously. No additional module or resource has been launched.

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

## Free-plan eligibility and timing caveat (checked 2026-10-24)

AWS currently lists `t3.micro`, `t3.small`, `t4g.micro`, `t4g.small`,
`c7i-flex.large`, and `m7i-flex.large` for new-account Free-plan EC2 eligibility.
`m7a.medium` is absent from that list; do not assume a price estimate for it can
be launched under the Free plan. Eligibility does not promise unlimited free
compute or available quota. Confirm the actual account plan, credits, region,
allowed types and quotas before choosing a request or revised estimate.
No account upgrade or instance type has been approved. The six-job serial pilot
has a $10 usage allowance; its exact resource/rate/cleanup plan remains pending.

`m7i-flex` provides a 40% baseline per vCPU and can reach full performance a
majority of the time. Do not silently substitute it for a sustained-performance
timing host. An eligible Flex instance may be considered for an approved
build/correctness smoke pilot, with timings labeled exploratory. Official
runtime comparisons require separate performance validation and host approval.
The eligible T-series choices also require attention to memory and CPU-credit
behavior; no candidate has been selected automatically. A revised estimate must
use the finally approved account plan/type rather than assume an upgrade.

Sources: [EC2 Free Tier eligibility](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-free-tier-usage.html),
[AWS free compute](https://aws.amazon.com/free/compute/), and
[M7i-flex baseline performance](https://aws.amazon.com/ec2/instance-types/m7i/).

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
