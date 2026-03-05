import torch
import torchvision.transforms as T
import timm
from lightglue import LightGlue, ALIKED

class ModelLoader:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._global_model = None
        self._local_extractor = None
        self._local_matcher = None

        self.global_transforms = T.Compose([
            T.Resize((384, 384)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @property
    def global_model(self):
        if self._global_model is None:
            print("Loading MegaDescriptor-L-384...")
            model = timm.create_model('hf_hub:BVRA/MegaDescriptor-L-384', pretrained=True, num_classes=0)
            model = model.to(self.device).eval()

            # Try to compile if supported
            try:
                model = torch.compile(model)
                print("Global model successfully compiled with torch.compile.")
            except Exception as e:
                print(f"Warning: torch.compile failed or unsupported on this platform: {e}")

            self._global_model = model
        return self._global_model

    @property
    def local_extractor(self):
        if self._local_extractor is None:
            print("Loading ALIKED extractor...")
            model = ALIKED(max_num_keypoints=2048).eval().to(self.device)
            self._local_extractor = model
        return self._local_extractor

    @property
    def local_matcher(self):
        if self._local_matcher is None:
            print("Loading LightGlue matcher...")
            model = LightGlue(features='aliked').eval().to(self.device)
            self._local_matcher = model
        return self._local_matcher

# Singleton instance
models = ModelLoader()
