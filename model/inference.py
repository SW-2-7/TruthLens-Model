# inference.py

from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms
import torch
import torch.nn as nn

from model.model import create_model


class DeepfakeDetector:
    def __init__(self, device: str = None):
        """Initializes the detector with MTCNN and Ensemble Models."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[System] Initializing Deepfake Detector on Device: {self.device}")

        # 1. Face Extractor (MTCNN)
        self.extractor = MTCNN(keep_all=False, post_process=False, device=self.device)

        # 2. Production Ensemble Models (EfficientNet-B0 + MobileNet-V3)
        self.model_eff = self._load_model("efficientnet_b0", "weights/dfdc_efficientnet_b0_focal.pth")
        self.model_mob = self._load_model("mobilenet_v3", "weights/dfdc_mobilenet_v3_focal.pth")
        
        # 3. Input Image Preprocessing Pipelines
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("[System] All deep learning models loaded successfully! 🚀")

    def _load_model(self, arch: str, path: str) -> nn.Module:
        """Helper function to cleanly parse and load state_dict layers."""
        model = create_model(arch=arch, num_classes=2)
        state = torch.load(path, map_location=self.device, weights_only=True)
        
        if "state_dict" in state: 
            state = state["state_dict"]
            
        # Strip DataParallel 'module.' prefixes if present
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        return model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        """Executes full inference pipeline: Face Detection -> Ensemble Scoring."""
        try:
            # 1. Load Raw Image Target
            img = Image.open(image_path).convert("RGB")
            
            # 2. Extract Facial Region of Interest (ROI)
            face = self.extractor(img)
            if face is None:
                return {"status": "error", "message": "No face detected in the target image."}
            
            # 🌟 [Bug Fix] Ensure tensor is safely copied to CPU before converting to numpy
            face_np = face.cpu().permute(1, 2, 0).numpy().astype('uint8')
            face_img = Image.fromarray(face_np)
            
            # 3. Apply Transformations & Model Scaling
            x = self.transform(face_img).unsqueeze(0).to(self.device)

            # 4. Core Weighted Ensemble Logic (Weighted Ratio -> 8 : 2)
            p_eff = torch.softmax(self.model_eff(x), dim=1)[0, 1].item()
            p_mob = torch.softmax(self.model_mob(x), dim=1)[0, 1].item()
            
            final_prob = (0.8 * p_eff) + (0.2 * p_mob)
            
            return {
                "status": "success",
                "label": "FAKE" if final_prob >= 0.5 else "REAL",
                "score": round(final_prob * 100, 2),
                "details": {
                    "efficientnet_b0": round(p_eff, 4), 
                    "mobilenet_v3": round(p_mob, 4)
                }
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}