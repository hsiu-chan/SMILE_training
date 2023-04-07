import sys
from segment_anything import sam_model_registry

class SamModel:
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    device = "cuda"
    model_type = "default"
    def __init__(self,checkpoint):
        sys.path.append("..")
        self.sam_checkpoint=checkpoint
        
        #self.sam_checkpoint=sam_checkpoint
        
    
    def build(self):
        sys.path.append("..")
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)
        return sam
