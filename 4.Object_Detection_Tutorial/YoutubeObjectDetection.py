{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "fc3a1d25-90f9-4fe4-bb7d-950372ab2223",
   "metadata": {},
   "outputs": [],
   "source": [
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "96da624e-ee6d-441d-80c9-e2193607b39e",
   "metadata": {},
   "outputs": [],
   "source": [
    "from YoutubeObjectDetection import *"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "b81cf944-e367-4a2c-b048-86a97f17ea7c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fasterrcnn_resnet50_fpn\n"
     ]
    },
    {
     "ename": "AttributeError",
     "evalue": "'FasterRCNN' object has no attribute 'names'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "Input \u001b[0;32mIn [2]\u001b[0m, in \u001b[0;36m<cell line: 2>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m url \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mhttps://www.youtube.com/watch?v=cNr3r3nT8KE\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m----> 2\u001b[0m objectdection \u001b[38;5;241m=\u001b[39m \u001b[43mObjectDetection\u001b[49m\u001b[43m(\u001b[49m\u001b[43murl\u001b[49m\u001b[43m,\u001b[49m\u001b[43mmodel_name\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mfasterrcnn_resnet50_fpn\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m)\u001b[49m\n\u001b[1;32m      3\u001b[0m objectdection()\n",
      "File \u001b[0;32m/red/ufhpc/hityangsir/ImageProcessingTutorial/Object_Detection_Youtube/YoutubeObjectDetection.py:33\u001b[0m, in \u001b[0;36mObjectDetection.__init__\u001b[0;34m(self, url, model_name, out_file)\u001b[0m\n\u001b[1;32m     31\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mmodel_name \u001b[38;5;241m=\u001b[39m model_name\n\u001b[1;32m     32\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mmodel \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mload_model(model_name)\n\u001b[0;32m---> 33\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mclasses \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mmodel\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mnames\u001b[49m\n\u001b[1;32m     34\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mout_file \u001b[38;5;241m=\u001b[39m out_file\n\u001b[1;32m     35\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mdevice \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mcuda\u001b[39m\u001b[38;5;124m'\u001b[39m \u001b[38;5;28;01mif\u001b[39;00m torch\u001b[38;5;241m.\u001b[39mcuda\u001b[38;5;241m.\u001b[39mis_available() \u001b[38;5;28;01melse\u001b[39;00m \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mcpu\u001b[39m\u001b[38;5;124m'\u001b[39m\n",
      "File \u001b[0;32m/red/ufhpc/hityangsir/.conda/envs/pytorch_lightning/lib/python3.9/site-packages/torch/nn/modules/module.py:1185\u001b[0m, in \u001b[0;36mModule.__getattr__\u001b[0;34m(self, name)\u001b[0m\n\u001b[1;32m   1183\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m name \u001b[38;5;129;01min\u001b[39;00m modules:\n\u001b[1;32m   1184\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m modules[name]\n\u001b[0;32m-> 1185\u001b[0m \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mAttributeError\u001b[39;00m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;132;01m{}\u001b[39;00m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m object has no attribute \u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;132;01m{}\u001b[39;00m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;241m.\u001b[39mformat(\n\u001b[1;32m   1186\u001b[0m     \u001b[38;5;28mtype\u001b[39m(\u001b[38;5;28mself\u001b[39m)\u001b[38;5;241m.\u001b[39m\u001b[38;5;18m__name__\u001b[39m, name))\n",
      "\u001b[0;31mAttributeError\u001b[0m: 'FasterRCNN' object has no attribute 'names'"
     ]
    }
   ],
   "source": [
    "url = \"https://www.youtube.com/watch?v=cNr3r3nT8KE\"\n",
    "objectdection = ObjectDetection(url,model_name='fasterrcnn_resnet50_fpn')\n",
    "objectdection()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "ad17852b-789d-4cce-a679-3e97e88e4320",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.12.0\n"
     ]
    }
   ],
   "source": [
    "import torchvision\n",
    "print(torchvision.__version__)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "050f6b7f-6520-4194-9582-94412d6d4744",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Downloading: \"https://github.com/pytorch/vision/archive/main.zip\" to /home/hityangsir/.cache/torch/hub/main.zip\n"
     ]
    },
    {
     "ename": "ImportError",
     "evalue": "cannot import name 'get_model_weights' from 'torchvision.models' (/red/ufhpc/hityangsir/.conda/envs/pytorch_lightning/lib/python3.9/site-packages/torchvision/models/__init__.py)",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mImportError\u001b[0m                               Traceback (most recent call last)",
      "Input \u001b[0;32mIn [7]\u001b[0m, in \u001b[0;36m<cell line: 1>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[43mtorch\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mhub\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mhelp\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mpytorch/vision\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43mresnet18\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mforce_reload\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43;01mTrue\u001b[39;49;00m\u001b[43m)\u001b[49m)\n",
      "File \u001b[0;32m/red/ufhpc/hityangsir/.conda/envs/pytorch_lightning/lib/python3.9/site-packages/torch/hub.py:335\u001b[0m, in \u001b[0;36mhelp\u001b[0;34m(github, model, force_reload, skip_validation)\u001b[0m\n\u001b[1;32m    332\u001b[0m sys\u001b[38;5;241m.\u001b[39mpath\u001b[38;5;241m.\u001b[39minsert(\u001b[38;5;241m0\u001b[39m, repo_dir)\n\u001b[1;32m    334\u001b[0m hubconf_path \u001b[38;5;241m=\u001b[39m os\u001b[38;5;241m.\u001b[39mpath\u001b[38;5;241m.\u001b[39mjoin(repo_dir, MODULE_HUBCONF)\n\u001b[0;32m--> 335\u001b[0m hub_module \u001b[38;5;241m=\u001b[39m \u001b[43m_import_module\u001b[49m\u001b[43m(\u001b[49m\u001b[43mMODULE_HUBCONF\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mhubconf_path\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    337\u001b[0m sys\u001b[38;5;241m.\u001b[39mpath\u001b[38;5;241m.\u001b[39mremove(repo_dir)\n\u001b[1;32m    339\u001b[0m entry \u001b[38;5;241m=\u001b[39m _load_entry_from_hubconf(hub_module, model)\n",
      "File \u001b[0;32m/red/ufhpc/hityangsir/.conda/envs/pytorch_lightning/lib/python3.9/site-packages/torch/hub.py:76\u001b[0m, in \u001b[0;36m_import_module\u001b[0;34m(name, path)\u001b[0m\n\u001b[1;32m     74\u001b[0m module \u001b[38;5;241m=\u001b[39m importlib\u001b[38;5;241m.\u001b[39mutil\u001b[38;5;241m.\u001b[39mmodule_from_spec(spec)\n\u001b[1;32m     75\u001b[0m \u001b[38;5;28;01massert\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(spec\u001b[38;5;241m.\u001b[39mloader, Loader)\n\u001b[0;32m---> 76\u001b[0m \u001b[43mspec\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mloader\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mexec_module\u001b[49m\u001b[43m(\u001b[49m\u001b[43mmodule\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m     77\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m module\n",
      "File \u001b[0;32m<frozen importlib._bootstrap_external>:850\u001b[0m, in \u001b[0;36mexec_module\u001b[0;34m(self, module)\u001b[0m\n",
      "File \u001b[0;32m<frozen importlib._bootstrap>:228\u001b[0m, in \u001b[0;36m_call_with_frames_removed\u001b[0;34m(f, *args, **kwds)\u001b[0m\n",
      "File \u001b[0;32m~/.cache/torch/hub/pytorch_vision_main/hubconf.py:4\u001b[0m, in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[38;5;66;03m# Optional list of dependencies required by the package\u001b[39;00m\n\u001b[1;32m      2\u001b[0m dependencies \u001b[38;5;241m=\u001b[39m [\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mtorch\u001b[39m\u001b[38;5;124m\"\u001b[39m]\n\u001b[0;32m----> 4\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mtorchvision\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mmodels\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m get_model_weights, get_weight\n\u001b[1;32m      5\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mtorchvision\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mmodels\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01malexnet\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m alexnet\n\u001b[1;32m      6\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mtorchvision\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mmodels\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mconvnext\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m convnext_base, convnext_large, convnext_small, convnext_tiny\n",
      "\u001b[0;31mImportError\u001b[0m: cannot import name 'get_model_weights' from 'torchvision.models' (/red/ufhpc/hityangsir/.conda/envs/pytorch_lightning/lib/python3.9/site-packages/torchvision/models/__init__.py)"
     ]
    }
   ],
   "source": [
    "print(torch.hub.help('pytorch/vision', 'resnet18', force_reload=True))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e0db7c22-fff3-4fa2-80a3-87924ece5565",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3.9.12 ('pytorch_lightning')",
   "language": "python",
   "name": "python3912jvsc74a57bd0996a809ff9bf08d21e36704bc158be9901188020768bbdc338e7277ca3e59a05"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
