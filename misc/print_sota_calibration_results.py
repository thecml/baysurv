import pandas as pd
import paths as pt
from pathlib import Path
from utility.model import map_model_name

if __name__ == "__main__":
    path = Path.joinpath(pt.RESULTS_DIR, f"sota_results.csv")
    results = pd.read_csv(path)
    results = results.round(3)

    model_names = ["cox", "coxnet", "coxboost", "rsf", "dsm", "dcm", "baycox", "baymtlr"]
    dataset_names = ["METABRIC", "SEER", "SUPPORT", "MIMIC"]
    model_citations = ['\cite{cox_regression_1972}', '\cite{simon_regularization_2011}',
                       '\cite{hothorn_survival_2005}', '\cite{ishwaran_random_2008}',
                       '\cite{nagpal_deep_2021}', '\cite{nagpal_deep_cox_2021}',
                       '\cite{qi_using_2023}', '\cite{qi_using_2023}']

    for dataset_name in dataset_names:
        for index, (model_name, model_citation) in enumerate(zip(model_names, model_citations)):
            text = ""
            res = results.loc[(results['DatasetName'] == dataset_name) & (results['ModelName'] == model_name)]
            if res.empty:
                break
            ici = float(res['ICI'])
            d_calib = float(res['DCalib'])
            c_calib = float(res['CCalib'])
            km = float(res['KM'])
            if d_calib > 0.05:
                d_calib = str(d_calib) + r"$^{\ast}$"
            if c_calib > 0.05:
                c_calib = str(c_calib) + r"$^{\ast}$"
            if model_name in ["cox", "coxnet", "coxboost", "rsf", "dsm", "dcm"]:
                c_calib = "-"
            model_name = map_model_name(model_name)
            text += f"{model_name} {model_citation} & "
            text += f"{ici} & {d_calib} & {c_calib} & {km} \\\\"
            print(text)
        print()
