import glob
from pathlib import Path

save_pharse = "checkpoint_"
base_path = "/content/tts-engine/gdrive/My Drive/outdir"

class autoload_checkpoint:
        def __init__(self):
                list = (glob.glob(base_path+"/*"))
                print(list)
                file_names = [] 
                for file in list:
                        name = Path(file).stem
                        name = name.replace('/content/tts-engine/gdrive/My Drive/outdir/checkpoint_',"")
                        file_names.append(file)
                results = [int(i) for i in file_names]

                latest = max(results)
                self.latest = base_path + save_pharse + latest
                print("biggest file is " + self.latest)
