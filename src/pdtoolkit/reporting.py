"""Generate a self-contained, paginated validation report and supporting figures."""

import os
import tempfile
from html import escape
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pd-toolkit-matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sklearn.metrics import roc_curve

from .metrics import calibration_table

NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#087E8B")
PALE = colors.HexColor("#EDF4F8")


def make_figures(predictions, validation, output, config):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for (model, variant), group in predictions.groupby(["model", "variant"]):
        label = f"{model.replace('_', ' ')} / {variant}"
        curve = calibration_table(
            group.default_12m,
            group.predicted_pd,
            config.validation.calibration_bins,
            config.validation.confidence,
        )
        axes[0].plot(curve.mean_pd, curve.observed_rate, "o-", ms=3, label=label)
        if variant == "calibrated":
            axes[0].vlines(curve.mean_pd, curve.observed_low, curve.observed_high, alpha=0.35)
        if group.default_12m.nunique() == 2:
            fpr, tpr, _ = roc_curve(group.default_12m, group.predicted_pd)
            axes[1].plot(fpr, tpr, label=label)
    limit = max(0.1, float(predictions.predicted_pd.quantile(0.95)))
    axes[0].plot([0, limit], [0, limit], "--", color="grey")
    axes[0].set(
        xlabel="Mean predicted one-year PD",
        ylabel="Observed default rate",
        title="Calibration by quantile bins",
    )
    axes[1].plot([0, 1], [0, 1], "--", color="grey")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="Final test ROC")
    axes[1].legend(loc="lower right", fontsize=6)
    path = output / "calibration_roc.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for (model, variant), group in predictions.groupby(["model", "variant"]):
        if variant != "calibrated":
            continue
        annual = group.groupby("observation_year").agg(
            mean_pd=("predicted_pd", "mean"), observed=("default_12m", "mean")
        )
        axes[0].plot(annual.index, annual.mean_pd, "o-", label=model.replace("_", " "))
    actual = (
        predictions.drop_duplicates("borrower_id").groupby("observation_year").default_12m.mean()
    )
    axes[0].plot(actual.index, actual, "s--", color="black", label="Observed")
    axes[0].set(
        xlabel="Observation year", ylabel="Default rate / mean PD", title="Out-of-time cohorts"
    )
    axes[0].set_xticks(actual.index)
    axes[0].legend(fontsize=7)
    drift = validation["drift_psi"]
    axes[1].barh([x.replace("_", " ") for x in drift], list(drift.values()), color="#087E8B")
    axes[1].axvline(config.validation.psi_warning, color="orange", ls="--")
    axes[1].set(xlabel="PSI vs training reference", title="Portfolio stability")
    path2 = output / "temporal_drift.png"
    fig.savefig(path2, dpi=170)
    plt.close(fig)
    return path, path2


def create_report(
    path, config, validation, metadata, predictions, coefficients, comparisons, effects
):
    path = Path(path)
    figures = make_figures(predictions, validation, path.parent / "figures", config)
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(name="TitleCustom", fontSize=25, leading=29, textColor=NAVY, spaceAfter=18)
    )
    styles.add(
        ParagraphStyle(name="Subtitle", fontSize=12, leading=17, textColor=TEAL, spaceAfter=14)
    )
    styles.add(ParagraphStyle(name="SmallCustom", fontSize=8, leading=11, spaceAfter=5))
    styles.add(ParagraphStyle(name="CellCustom", fontSize=7.4, leading=10))
    styles.add(ParagraphStyle(name="Centre", alignment=TA_CENTER, fontSize=9, leading=12))
    styles["BodyText"].fontSize = 9.5
    styles["BodyText"].leading = 14
    styles["BodyText"].spaceAfter = 8
    story = []

    def p(text, style="BodyText"):
        story.append(Paragraph(escape(str(text)), styles[style]))

    def heading(number, title):
        p(f"{number:02d}  {title}", "Heading1")
        story.append(Spacer(1, 8))

    def table(rows, widths=None):
        wrapped = [
            [Paragraph(escape(str(cell)), styles["CellCustom"]) for cell in row] for row in rows
        ]
        obj = Table(wrapped, colWidths=widths, repeatRows=1, hAlign="LEFT")
        obj.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        # Paragraphs carry their own colour, so make header text white explicitly.
        for cell in wrapped[0]:
            cell.style = ParagraphStyle(
                name="Header", parent=styles["CellCustom"], textColor=colors.white
            )
        story.append(obj)
        story.append(Spacer(1, 12))

    def fmt(value, digits=3):
        return "Not estimable" if value is None else f"{value:.{digits}f}"

    def ci(row, key):
        values = row["confidence_intervals"][key]
        return "Not estimable" if values is None else f"[{values[0]:.3f}, {values[1]:.3f}]"

    overall = [r for r in validation["metrics"] if r["population"] == "overall"]
    p("CREDIT RISK / MODEL VALIDATION", "Subtitle")
    p("One-Year Probability\nof Default", "TitleCustom")
    p("Development and out-of-time validation toolkit", "Subtitle")
    p(f"Scenario: {config.data.scenario} | Seed: {config.data.seed} | As of: {config.data.as_of}")
    p(
        "SYNTHETIC DATA ONLY. This report is an engineering and statistical demonstration. "
        "It does not constitute banking production approval, regulatory compliance or a real borrower risk assessment."
    )
    heading(1, "Executive decision")
    p(validation["conclusion"], "Heading2")
    p(
        f"Final test: {len(predictions.drop_duplicates('borrower_id')):,} independent borrowers, "
        f"{int(predictions.drop_duplicates('borrower_id').default_12m.sum()):,} defaults. "
        "Decisions below apply to calibrated predictions and configured illustrative thresholds."
    )
    table(
        [["Model / variant", "AUC", "Gini", "Mean PD", "Observed", "Log loss"]]
        + [
            [
                f"{r['model']} / {r['variant']}",
                fmt(r["roc_auc"]),
                fmt(r["gini"]),
                f"{r['mean_pd']:.2%}",
                f"{r['observed_rate']:.2%}",
                fmt(r["log_loss"]),
            ]
            for r in overall
        ],
        [167, 50, 50, 70, 70, 68],
    )
    p(
        f"Findings: {sum(f['severity'] == 'HIGH' for f in validation['findings'])} high, "
        f"{sum(f['severity'] == 'MEDIUM' for f in validation['findings'])} medium, "
        f"{sum(f['severity'] == 'LIMITATION' for f in validation['findings'])} limitations."
    )
    p(
        "The demonstration retains both models. Final test results are not used to tune, recalibrate, "
        "refit or automatically select a deployment model. Remediation requires a new mature validation cohort."
    )
    story.append(PageBreak())
    heading(2, "Data, target and development boundaries")
    p(
        "A record is an independent borrower profile on 1 January. The target is a realised default "
        "strictly after observation and within the next twelve calendar months. Non-defaults also "
        "require the full outcome window to mature. Feature availability dates must be no later than observation."
    )
    table(
        [["Stage", "Observation years", "Borrowers", "Defaults", "Role"]]
        + [
            [
                name,
                ", ".join(map(str, getattr(config.splits, name))),
                detail["n"],
                detail["defaults"],
                {
                    "train": "Fit preprocessing and model",
                    "tune": "Select parameters by log loss",
                    "calibrate": "Fit probability mapping",
                    "test": "Final evaluation only",
                }[name],
            ]
            for name, detail in metadata["splits"].items()
        ],
        [67, 125, 65, 60, 158],
    )
    p(
        "Intervening years are deliberately unused. Each earlier stage's latest outcome date must "
        "precede the next stage's first prediction date. This prevents historical labels being available too early."
    )
    p(
        "Only the approved feature list reaches the pipeline. IDs, dates, realised targets and true PDs "
        "are excluded. True PDs are in oracle.csv and enter benchmark calculations only after fitting is complete."
    )
    table(
        [
            ["Integrity control", "Result"],
            ["Feature availability and allowlist", "Passed"],
            ["Unique borrower IDs / binary target / event dates", "Passed"],
            ["Twelve-month horizons and complete maturity", "Passed"],
            ["Separate training, tuning, calibration and final test", "Passed"],
            ["Missing values", "Measured; train-fitted median imputation with missing indicators"],
        ],
        [235, 240],
    )
    p(
        "Model parameters and calibration details are recorded in metadata.json. Integrity defects "
        "abort modelling and produce integrity_failure.json instead of a misleading validation report."
    )
    story.append(PageBreak())
    heading(3, "Discrimination and probability calibration")
    story.append(Image(str(figures[0]), width=6.6 * inch, height=2.38 * inch))
    story.append(Spacer(1, 10))
    table(
        [["Model / variant", "AUC confidence interval", "Brier", "Slope", "Intercept"]]
        + [
            [
                f"{r['model']} / {r['variant']}",
                ci(r, "roc_auc"),
                fmt(r["brier"], 4),
                fmt(r["calibration_slope"]),
                fmt(r["calibration_intercept"]),
            ]
            for r in overall
        ],
        [164, 120, 65, 63, 63],
    )
    p(
        "Gini = 2 x AUC - 1. Brier score and log loss use realised binary outcomes; lower values are "
        "better. Calibration slope and intercept are diagnostic fits on test labels, not an applied test-time recalibration."
    )
    p(
        f"Confidence level: {config.validation.confidence:.0%}; {config.validation.bootstrap_samples} "
        "ordinary borrower bootstrap replicates. Single-class replicates are omitted only for AUC/Gini. "
        "These intervals condition on the simulated macro paths and fitted models; they exclude refitting "
        "uncertainty, future macro uncertainty and correlated real-world defaults."
    )
    p(
        "Vertical lines show Wilson intervals for observed bin rates. Quantile bins are descriptive "
        "and may collapse when predictions are tied. Calibration alarms additionally require a material "
        "gap and a gap outside the independent-Bernoulli null interval. Subgroup flags are exploratory; "
        "multiple-testing correction is not applied."
    )
    if comparisons:
        comparison = comparisons.get("calibrated", {})
        table(
            [["Paired challenger minus baseline", "Difference", "Confidence interval"]]
            + [
                [
                    key,
                    fmt(value["challenger_minus_baseline"], 4),
                    "Not estimable"
                    if value["ci"] is None
                    else str([round(x, 4) for x in value["ci"]]),
                ]
                for key, value in comparison.items()
            ],
            [220, 95, 160],
        )
    story.append(PageBreak())
    heading(4, "Temporal and population stability")
    story.append(Image(str(figures[1]), width=6.6 * inch, height=2.31 * inch))
    story.append(Spacer(1, 10))
    grouped = [
        r
        for r in validation["metrics"]
        if r["variant"] == "calibrated" and r["population"] != "overall"
    ]
    table(
        [["Model", "Population", "N / defaults", "AUC", "PD / observed"]]
        + [
            [
                r["model"],
                f"{r['population']}: {r['label']}",
                f"{r['n']} / {r['defaults']}",
                fmt(r["roc_auc"]),
                f"{r['mean_pd']:.2%} / {r['observed_rate']:.2%}",
            ]
            for r in grouped
        ],
        [112, 122, 82, 60, 99],
    )
    p(
        "PSI uses training quantile bins, a separate missing bin and 0.5-count smoothing. Constant "
        "training variables retain two tail bins so out-of-range shifts remain visible. PSI is a "
        "descriptive drift measure and its threshold is not a hypothesis test."
    )
    p(
        "Downturn, portfolio composition and mechanism-shift scenarios alter final test cohorts "
        "only. Development data remain identical across these scenarios for a fixed seed. "
        "Low-default segment results may have wide intervals even when aggregate metrics look stable."
    )
    story.append(PageBreak())
    heading(5, "Conceptual soundness and benchmark")
    p(
        "Default generation uses a logistic link with income, DTI, utilisation, delinquency, loan size, "
        "segment and contemporaneous macro conditions; it adds nonlinear utilisation and delinquency-DTI "
        "interactions. Income is protective; other borrower risk factors increase PD. Unemployment "
        "increases PD and GDP growth reduces it. Features share a latent borrower-risk factor."
    )
    p(
        "Logistic regression offers transparent conditional coefficients but omits nonlinear terms. "
        "Histogram gradient boosting captures interactions with additional complexity. Both models use "
        "identical train-fitted preprocessing. Model and calibrator selection are predeclared; no final-test optimisation occurs."
    )
    if not coefficients.empty:
        main = coefficients.loc[
            (coefficients.window == "all_train")
            & ~coefficients.feature.str.startswith("missingindicator")
        ]
        table(
            [["Logistic training variable", "Standardised coefficient"]]
            + [[r.feature, f"{r.coefficient:.3f}"] for r in main.itertuples()],
            [300, 175],
        )
    p(
        "Numeric coefficients apply to standardised, imputed inputs. Segment coefficients are relative "
        "to the alphabetically first category. Annual coefficient stability and train-only perturbation "
        "sensitivities are exported. These are associations, not causal effects; correlated macro series "
        "and sparse annual defaults can make individual coefficients unstable."
    )
    benchmarks = metadata["oracle_benchmark"]
    table(
        [["Post-fit benchmark", "Value"]]
        + [
            ["Oracle expected default rate", f"{benchmarks['mean_true_pd']:.2%}"],
            ["Oracle realised-outcome Brier", f"{benchmarks['oracle_brier']:.4f}"],
            *[
                [f"{name}: mean squared PD error vs oracle", f"{value:.6f}"]
                for name, value in benchmarks["pd_mse"].items()
            ],
        ],
        [320, 155],
    )
    story.append(PageBreak())
    heading(6, "Findings, affected populations and actions")
    ordered = sorted(
        validation["findings"],
        key=lambda f: {"HIGH": 0, "MEDIUM": 1, "LIMITATION": 2}[f["severity"]],
    )
    if not ordered:
        p(
            "No material issue was identified by the configured checks. This is limited evidence "
            "within a synthetic experiment and must not be read as assurance of production suitability."
        )
    for finding in ordered[:8]:
        p(
            f"{finding['severity']} / {finding['code']} / {finding['model']} / {finding['population']}",
            "Heading3",
        )
        p(finding["message"] + " Action: " + finding["action"], "SmallCustom")
    if len(ordered) > 8:
        p(
            f"{len(ordered) - 8} additional findings are available in findings.csv and validation.json."
        )
    p("Limitations", "Heading2")
    p(
        "Annual cohorts contain independent synthetic borrowers, not repeated account histories. "
        "Macro paths are stylised rather than forecasts. Missingness is mildly risk dependent; survival "
        "censoring, prepayment, recoveries, selection bias, borrower contagion and intervention effects "
        "are not modelled. This is point-in-time one-year PD, not long-run regulatory PD, LGD, EAD or IFRS 9."
    )
    p(
        "Thresholds are configurable demonstrations, not bank policy. Calibration may not improve "
        "every metric or scenario. Performance gains must be weighed against interpretability, default "
        "counts and stability. Remedies should be tested on new mature data before expanding use."
    )
    story.append(PageBreak())
    heading(7, "Reproducibility and evidence")
    table(
        [
            ["Item", "Recorded value"],
            ["Toolkit version", metadata["toolkit_version"]],
            ["Python", metadata["python"]],
            ["Seed", config.data.seed],
            ["Git revision", metadata["git_revision"] or "No Git revision available"],
            ["Working tree modified", metadata["git_dirty"]],
            ["Source fingerprint", metadata["source_sha256"]],
            ["Observable data fingerprint", metadata["data_sha256"]],
        ],
        [150, 325],
    )
    p(
        "The evidence bundle includes observable data, availability metadata, a separate oracle, "
        "all final predictions, calibration bins, overall/period/segment metrics, paired comparisons, "
        "findings, tuning results, coefficients, sensitivities, fitted models, exact resolved configuration, "
        "dependency versions and a SHA-256 artifact manifest. Run pd-toolkit verify to validate file integrity."
    )
    p(
        "Identical configuration, source and dependency lock should reproduce predictions and metrics "
        "within absolute tolerance 1e-10 on the same platform. PDF creation dates, run timestamps, "
        "runtime duration and platform-specific numerical differences are outside the bit-for-bit guarantee."
    )
    p("Methodology references", "Heading2")
    p(
        "scikit-learn: Probability calibration. https://scikit-learn.org/stable/modules/calibration.html",
        "SmallCustom",
    )
    p(
        "Basel Committee (2005): Studies on the Validation of Internal Rating Systems. "
        "https://www.bis.org/publ/bcbs_wp14.pdf",
        "SmallCustom",
    )
    p(
        "Basel Committee (2005): Validation of low-default portfolios. "
        "https://www.bis.org/publ/bcbs_nl6.pdf",
        "SmallCustom",
    )
    p(
        "These references inform the distinction between discrimination, calibration and sparse-default "
        "limitations. Referencing them does not imply regulatory compliance.",
        "SmallCustom",
    )

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(TEAL)
        canvas.line(38, 42, 557, 42)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(NAVY)
        canvas.drawString(38, 28, "PD VALIDATION TOOLKIT | SYNTHETIC DATA")
        canvas.drawRightString(557, 28, str(document.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(path),
        pagesize=(595.28, 841.89),
        leftMargin=40,
        rightMargin=40,
        topMargin=42,
        bottomMargin=56,
        title="One-Year PD Validation Report",
        author="PD Validation Toolkit",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return path
