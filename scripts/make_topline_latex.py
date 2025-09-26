# save as scripts/make_topline_latex.py (or run in a python - <<'PY' block)
import pandas as pd, os
p = "data/derived/two_track_topline.csv"
df = pd.read_csv(p)

# 보기 좋게 포맷
fmt = df.copy()
fmt["MAE"]        = fmt["MAE"].map(lambda x: f"{x:.3f}")
fmt["MdAE"]       = fmt["MdAE"].map(lambda x: f"{x:.3f}")
fmt["RMSE"]       = fmt["RMSE"].map(lambda x: f"{x:.3f}")
fmt["coverage"]   = fmt["coverage"].map(lambda x: f"{100*x:.1f}%")
fmt["mean_width"] = fmt["mean_width"].map(lambda x: f"{x:.2f}")
if "retention" in fmt.columns:
    fmt["retention"] = fmt["retention"].map(lambda x: f"{100*x:.1f}%")

cols = ["label","n","MAE","MdAE","RMSE","coverage","mean_width"] + (["retention"] if "retention" in fmt.columns else [])
fmt = fmt[cols].rename(columns={
    "label":"Track",
    "n":"N",
    "MAE":"MAE (bpm)",
    "MdAE":"MdAE (bpm)",
    "RMSE":"RMSE (bpm)",
    "coverage":"PI cov.",
    "mean_width":"PI width (bpm)",
    "retention":"Retention"
})

latex = fmt.to_latex(index=False, escape=False, longtable=False, bold_rows=False,
                     column_format="lrrrrrr" + ("r" if "Retention" in fmt.columns else ""))

os.makedirs("data/derived", exist_ok=True)
with open("data/derived/two_track_topline.tex","w") as f:
    f.write("\\begin{table}[t]\n\\centering\n")
    f.write("\\caption{Two-track summary (All-comers vs High-confidence gate).}\n")
    f.write("\\label{tab:two_track_topline}\n")
    f.write(latex)
    f.write("\n\\end{table}\n")
print("Wrote: data/derived/two_track_topline.tex")
