from ultralytics.data.annotator import auto_annotate

auto_annotate(data=r"test", det_model="best.pt", sam_model=r'sam_b.pt',device='cuda')