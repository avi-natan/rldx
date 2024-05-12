import os
import subprocess


def submit_job(filename):
    with open(sbatch_file_name, 'w+') as f:
        f.write(f'#!/bin/bash\n')
        f.write(f'\n')
        f.write(f'################################################################################################\n')
        f.write(f'### sbatch configuration parameters must start with #SBATCH and must precede any other commands.\n')
        f.write(f'### To ignore, just add another # - like so: ##SBATCH\n')
        f.write(f'################################################################################################\n')
        f.write(f'\n')
        f.write(f'#SBATCH --partition main			### specify partition name where to run a job. short: 7 days limit; gtx1080: 7 days; debug: 2 hours limit and 1 job at a time\n')
        f.write(f'#SBATCH --time 6-23:30:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS\n')
        f.write(f'#SBATCH --job-name {filename}			### name of the job\n')
        f.write(f'#SBATCH --output logs/{filename}-job-%J.out			### output log for running job - %J for job number\n')
        f.write(f'#SBATCH --ntasks=1\n')
        f.write(f'#SBATCH --cpus-per-task=4\n')
        f.write(f'#SBATCH -D /home/natanavr/natanavr/rldx\n')
        f.write(f'\n')
        f.write(f'## Note: the following 4 lines are commented out\n')
        f.write(f'#SBATCH --mail-user=natanavr@post.bgu.ac.il	### users email for sending job status messages\n')
        f.write(f'#SBATCH --mail-type=END,FAIL			### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE\n')
        f.write(f'##SBATCH --mem=32G				### ammount of RAM memory\n')
        f.write(f'\n')
        f.write(f'### Print some data to output file ###\n')
        f.write(f'echo `date`\n')
        f.write(f'echo -e "\\nSLURM_JOBID:\\t\\t" $SLURM_JOBID\n')
        f.write(f'echo -e "SLURM_JOB_NODELIST:\\t" $SLURM_JOB_NODELIST "\\n\\n"\n')
        f.write(f'\n')
        f.write(f'### Start your code below ####\n')
        f.write(f'module load anaconda				### load anaconda module (must be present when working with conda environments)\n')
        f.write(f'source activate rldx3_8				### activate a conda environment, replace my_env with your conda environment\n')
        f.write(f'python main.py {filename}.json				### execute python script – replace with your own command \n')

    data = subprocess.check_output(['sbatch', sbatch_file_name]).decode()
    return int(data.split()[-1])

def main():
    filenames = os.listdir("experimental inputs")
    for filename in filenames:
        submit_job(filename[:-5])


if __name__ == '__main__':
    sbatch_file_name = 'rldx_temp.example'
    main()
