import os

# Ensure directory exists
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def GenerateGitPatchAndLog(logFileName,GitPatchName):
    #CMSSWDirPath = os.environ['CMSSW_BASE']
    #CMSSWRel = CMSSWDirPath.split("/")[-1]

    os.system('git diff > '+GitPatchName)

    outScript = open(logFileName,"w");
    #outScript.write('\nCMSSW Version used: '+CMSSWRel+'\n')
    #outScript.write('\nCurrent directory path: '+CMSSWDirPath+'\n')
    outScript.close()

    os.system('echo -e "\n\n============\n== Latest commit summary \n\n" >> '+logFileName )
    os.system("git log -1 --pretty=tformat:' Commit: %h %n Date: %ad %n Relative time: %ar %n Commit Message: %s' >> "+logFileName )
    os.system('echo -e "\n\n============\n" >> '+logFileName )
    os.system('git log -1 --format="SHA: %H" >> '+logFileName )
