import glob
from pathlib import Path

save_pharse = "checkpoint_"
base_path = "/content/tts-engine/gdrive/My Drive/outdir"

class autoload_checkpoint:
        def __init__(self):
                list = glob.glob("/content/tts-engine/gdrive/My Drive/outdir/checkpoint_*")
                print(list)
                file_names = [] 
                for file in list:
                        name = file.replace('/content/tts-engine/gdrive/My Drive/outdir/checkpoint_',"")
                        file_names.append(name)
                results = [int(i) for i in file_names]

                latest = max(results)
                self.latest = base_path +"/"+ save_pharse + str(latest)
                print("biggest file is " + self.latest)
