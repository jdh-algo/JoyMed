import os
from utils import (
    Medbullets_op4,
    Medbullets_op5,
    MedXpertQA,
    MMMU,
    OmniMedVQA,
    PATH_VQA,
    PMC_VQA,
    SLAKE,
    SuperGPQA,
    VQA_RAD,
    HealthBench,
    PubMedQA,
    MedMCQA,
    MedQA_USMLE,
    MMLU,
    CMB,
    CMExam,
    MedQA_MCMLE,
    CMMLU,
    IU_XRAY,
    CheXpert_Plus,
    MIMIC_CXR,
    MedFrameQA,
    DiagnosisArena,
    GMAIMMBench,
    CT_RATE,
    M3D,
    RAD_3D,
    RAD_3D_ori,
    MIR_3D,
    AMOS,
    AMOS_ori,
    INSPECT,
    Deeptumor,
    Know2Do,
    Professional,
    )
#eval_MedQA_USMLE
def prepare_benchmark(model,eval_dataset,eval_dataset_path,eval_output_path):
    # Hallu,Geometry3k
    supported_dataset = ["MMMU-Medical-test","MMMU-Medical-val","PATH_VQA","PMC_VQA","VQA_RAD","SLAKE","MedQA_USMLE","MedMCQA","PubMedQA","OmniMedVQA",
                         "Medbullets_op4","Medbullets_op5","MedXpertQA-Text","MedXpertQA-MM","SuperGPQA","HealthBench",
                         "IU_XRAY","CheXpert_Plus",
                         "MIMIC_CXR","CMB","CMExam","CMMLU","MedQA_MCMLE","MedFrameQA",
                         "DiagnosisArena","GMAI-MMBench-val","GMAI-MMBench-test",
                         "Know2Do","Professional",
                         "CT-RATE-Report","CT-RATE-VQA","M3D-Ori","M3D-Slice","3D-RAD","3D-RAD-Ori","3D-MIR","INSPECT","AMOS-MM-VQA","AMOS-MM-Report","AMOS-Ori-Report","Deeptumor-VQA","Deeptumor-Report"]

    if eval_dataset in ["MMMU-Medical-test", "MMMU-Medical-val"]:
        if eval_dataset_path:
            eval_dataset_path = eval_dataset_path.replace(eval_dataset,"MMMU")
        _ , subset , split = eval_dataset.split("-")
        dataset = MMMU(model,eval_dataset_path,eval_output_path,split,subset)

    elif eval_dataset == "PATH_VQA":
        dataset = PATH_VQA(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "PMC_VQA":
        dataset = PMC_VQA(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "VQA_RAD":
        dataset = VQA_RAD(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "SLAKE":
        dataset = SLAKE(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "OmniMedVQA":
        dataset = OmniMedVQA(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "Medbullets_op4":
        dataset = Medbullets_op4(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "Medbullets_op5":
        dataset = Medbullets_op5(model,eval_dataset_path,eval_output_path)

    elif eval_dataset in ["MedXpertQA-Text","MedXpertQA-MM"]:
        if eval_dataset_path:
            eval_dataset_path = eval_dataset_path.replace(eval_dataset,"MedXpertQA")
        _,split = eval_dataset.split("-")
        dataset = MedXpertQA(model,eval_dataset_path,eval_output_path,split)

    elif eval_dataset == "SuperGPQA":
        dataset = SuperGPQA(model,eval_dataset_path,eval_output_path)
    
    elif eval_dataset == "HealthBench":
        dataset =HealthBench(model,eval_dataset_path,eval_output_path)
    
    elif eval_dataset == "PubMedQA":
        dataset = PubMedQA(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "MedMCQA":
        dataset = MedMCQA(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "MedQA_USMLE":
        dataset = MedQA_USMLE(model,eval_dataset_path,eval_output_path)
    
    elif eval_dataset in ["MMLU-medical","MMLU-all"]:
        if eval_dataset_path:
            eval_dataset_path = eval_dataset_path.replace(eval_dataset,"MMLU")
        _,subject = eval_dataset.split("-")
        dataset = MMLU(model,eval_dataset_path,eval_output_path,subject)
    
    elif eval_dataset == "CMB":
        dataset = CMB(model,eval_dataset_path,eval_output_path)
    
    elif eval_dataset == "CMExam":
        dataset = CMExam(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "MedQA_MCMLE":
        dataset = MedQA_MCMLE(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "CMMLU":
        dataset = CMMLU(model,eval_dataset_path,eval_output_path)
    
    elif eval_dataset == "IU_XRAY":
        dataset = IU_XRAY(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "CheXpert_Plus":
        dataset = CheXpert_Plus(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "MIMIC_CXR":
        dataset = MIMIC_CXR(model,eval_dataset_path,eval_output_path)
    
    elif eval_dataset == "MedFrameQA":
        dataset = MedFrameQA(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "DiagnosisArena":
        dataset = DiagnosisArena(model,eval_dataset_path,eval_output_path)
        
    elif eval_dataset in ["GMAI-MMBench-test", "GMAI-MMBench-val"]:
        dataset = GMAIMMBench(model,eval_dataset_path,eval_output_path)
        
    elif eval_dataset in ["CT-RATE-Report","CT-RATE-VQA"]:
        if eval_dataset_path:
            eval_dataset_path = eval_dataset_path.replace(eval_dataset,"CT-RATE")
        task = eval_dataset.split('-')[-1]
        dataset = CT_RATE(model,eval_dataset_path,eval_output_path, task)
        
    elif eval_dataset in ["M3D-Ori","M3D-Slice"]:
        if eval_dataset_path:
            eval_dataset_path = eval_dataset_path.replace(eval_dataset,"M3D")
        mode = eval_dataset.split('-')[-1]
        dataset = M3D(model,eval_dataset_path,eval_output_path, mode)
        
    elif eval_dataset == "3D-RAD":
        dataset = RAD_3D(model,eval_dataset_path,eval_output_path)

    elif eval_dataset == "3D-RAD-Ori":
        dataset = RAD_3D_ori(model,eval_dataset_path,eval_output_path)
        
    elif eval_dataset == "3D-MIR":
        dataset = MIR_3D(model,eval_dataset_path,eval_output_path)
        
    elif eval_dataset == "INSPECT":
        dataset = INSPECT(model,eval_dataset_path,eval_output_path)
        
    elif eval_dataset in ["Deeptumor-Report","Deeptumor-VQA"]:
        if eval_dataset_path:
            eval_dataset_path = eval_dataset_path.replace(eval_dataset,"Deeptumor")
        task = eval_dataset.split('-')[-1]
        dataset = Deeptumor(model,eval_dataset_path,eval_output_path, task)
        
    elif eval_dataset in ["AMOS-MM-VQA","AMOS-MM-Report"]:
        if eval_dataset_path:
            eval_dataset_path = eval_dataset_path.replace(eval_dataset,"AMOS")
        _,split,task = eval_dataset.split("-")
        dataset = AMOS(model,eval_dataset_path,eval_output_path,task,split)

    elif eval_dataset == "AMOS-Ori-Report":
        if eval_dataset_path:
            eval_dataset_path = eval_dataset_path.replace(eval_dataset, os.path.join("AMOS", "ORI"))
        dataset = AMOS_ori(model,eval_dataset_path,eval_output_path)
        
    elif eval_dataset == "Know2Do":
        dataset = Know2Do(model,eval_dataset_path,eval_output_path)
        
    elif eval_dataset.startswith("Professional"):
        if eval_dataset_path:
            eval_dataset_path = eval_dataset_path.replace(eval_dataset,"Professional")
        task = eval_dataset.split('-')[-1]
        dataset = Professional(model,eval_dataset_path,eval_output_path, task)
        
    else:
        print(f"unknown eval dataset {eval_dataset}, we only support {supported_dataset}")
        dataset = None
        

    return dataset

if __name__ == '__main__':
    pass    