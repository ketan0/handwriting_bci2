#!/usr/bin/env python3
import subprocess

import numpy as np

class TuneUniform:
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def sample(self):
        return np.random.uniform(self.start, self.end)

class TuneLogUniform:
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def sample(self):
        return np.exp(np.random.uniform(
            np.log(self.start), np.log(self.end)
        ))

class TuneChoice:
    def __init__(self, choices):
        self.choices = choices
    def sample(self):
        return np.random.choice(self.choices)

def submit_job(wrap_cmd, job_name='sbatch', mail_type=None,
               mail_user=None, p='normal,hns', c=1, t=2, **kwargs):
    """submit_job: Wrapper to submit sbatch jobs to Slurm.
    :param wrap_cmd: command to execute in the job.
    :param job_name: name for the job.
    :param mail_type: mail upon success or fail.
    :param mail_user: user email.
    :param p: partitions to select from.
    :param c: Number of cores to use.
    :param t: Time to run the job for.
    :param **kwargs: Additional command-line arguments to sbatch.
    See https://www.sherlock.stanford.edu/docs/getting-started/submitting/
    """
    def _job_time(t):
        """_job_time: Converts time t to slurm time ('hh:mm:ss').
        :param t: a float representing # of hours for the job.
        """
        hrs = int(t // 1)
        mins = int(t * 60 % 60)
        secs = int(t * 3600 % 60)

        return f'{str(hrs).zfill(2)}:{str(mins).zfill(2)}:{str(secs).zfill(2)}'

    args = []
    args.extend(['-p', str(p)])
    args.extend(['-c', str(c)])
    args.extend(['-t', _job_time(t)])
    args.extend(['--job-name', job_name])
    if mail_type:
        args.extend(['--mail-type', mail_type])
    if mail_user:
        args.extend(['--mail-user', mail_user])

    for opt, optval in kwargs.items():
        args.extend(['--' + opt, optval])
    args.extend(['--wrap', wrap_cmd])

    p = subprocess.Popen(['sbatch'] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.communicate()

if __name__ == '__main__':
    hparams = {
        'lr': TuneLogUniform(1e-5, 1e-2),
        'weight_decay': TuneUniform(0, 0.3),
        'dropout': TuneUniform(0, 0.3)
    }
    n_samples = 20
    for i in range(n_samples):
        hparams_i = {k:v.sample() for k, v in hparams.items()}
        print(f'Submitting job with hparams: {hparams_i}')
        cmd = ('./run_all_days_conformer_slurm_hyperparam_tune.sh '
               f'{hparams_i["lr"]} {hparams_i["weight_decay"]} {hparams_i["dropout"]}')
        result = submit_job(cmd, job_name=f'sample_{i}', p='gpu', t=5.0, mem='32G',
                            gres='gpu:1', constraint='GPU_MEM:32GB')
        print(result)
