#!/usr/bin/env python3
"""
Generate personalized ClimateVision role assignment PDFs for each team member.
"""

from fpdf import FPDF
import os

OUTPUT_DIR = "/Users/starrexshotit/Desktop/ClimateVision-main/team_docs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class RoleDoc(FPDF):
    def __init__(self, member_name):
        super().__init__()
        self.member_name = member_name

    def header(self):
        # Green header bar
        self.set_fill_color(34, 120, 74)
        self.rect(0, 0, 210, 28, 'F')
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(255, 255, 255)
        self.set_y(5)
        self.cell(0, 10, "ClimateVision", align="L", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 6, "Role Assignment & Codebase Ownership", align="L", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 10, f"ClimateVision | Confidential - Prepared for {self.member_name} | Page {self.page_no()}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(34, 120, 74)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        # Underline
        self.set_draw_color(34, 120, 74)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def subsection_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)
        self.set_text_color(0, 0, 0)

    def _sanitize(self, text):
        """Replace unicode chars that latin-1 can't handle."""
        replacements = {
            '\u2013': '-',   # en dash
            '\u2014': '-',   # em dash
            '\u2018': "'",   # left single quote
            '\u2019': "'",   # right single quote
            '\u201c': '"',   # left double quote
            '\u201d': '"',   # right double quote
            '\u2022': '-',   # bullet
            '\u2026': '...', # ellipsis
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, self._sanitize(text))
        self.ln(2)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        x = self.get_x()
        self.cell(6, 5.5, "-", new_x="END")
        self.multi_cell(0, 5.5, self._sanitize(text))
        self.ln(1)

    def code_block(self, text):
        self.set_font("Courier", "", 9)
        self.set_fill_color(240, 240, 240)
        lines = text.strip().split("\n")
        for line in lines:
            self.cell(0, 5, "  " + line, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)
        self.set_font("Helvetica", "", 10)

    def key_value(self, key, value):
        self.set_font("Helvetica", "B", 10)
        self.cell(45, 6, self._sanitize(key) + ":", new_x="END")
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 6, self._sanitize(value))
        self.ln(1)

    def month_block(self, month_title, weeks):
        self.set_font("Helvetica", "B", 10)
        self.set_fill_color(34, 120, 74)
        self.set_text_color(255, 255, 255)
        self.cell(0, 7, "  " + month_title, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(2)
        for week_title, tasks in weeks:
            self.set_font("Helvetica", "B", 10)
            self.cell(0, 6, week_title, new_x="LMARGIN", new_y="NEXT")
            self.ln(1)
            for task in tasks:
                self.bullet(task)
        self.ln(2)


def create_adeolu_doc():
    pdf = RoleDoc("Adeolu Mary Oshadare")
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Adeolu Mary Oshadare", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7, "Data Science Engineer 2 - Data Pipeline & GIS Lead", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # Quick Info
    pdf.key_value("GitHub", "@Oshgig")
    pdf.key_value("Access Level", "Maintainer")
    pdf.key_value("Reports To", "@Goldokpa (Project Owner)")
    pdf.key_value("Project Duration", "3 Months")
    pdf.ln(3)

    # How It Fits Me
    pdf.section_title("How This Role Fits You")
    pdf.body_text(
        "Your B.Tech in Remote Sensing & GIS from FUTA gives you something no one else on this team has - "
        "a formal education in exactly the kind of spatial data ClimateVision processes. You understand "
        "satellite imagery at a fundamental level: spectral bands, atmospheric correction, spatial resolution, "
        "and coordinate reference systems."
    )
    pdf.body_text(
        "As a GIS Analyst at Charis Tech Hub, you already worked with Google Earth Engine and AWS, writing "
        "Python scripts to model and extract insights from large geospatial datasets. That is precisely what "
        "ClimateVision's data pipeline needs - someone who can build the bridge between raw Sentinel-2 imagery "
        "and the clean, preprocessed tensors our ML models consume."
    )
    pdf.body_text(
        "Your MSc in Data Science from Hertfordshire added the machine learning layer: Scikit-Learn, TensorFlow, "
        "XGBoost, Pandas, and data pipelines. Your credit card fraud detection project showed you can handle "
        "imbalanced datasets (SMOTE) and build production-quality ML models - the same skills needed when dealing "
        "with satellite imagery where cloud-free forest pixels are the minority class."
    )
    pdf.body_text(
        "Your experience with Power BI, Tableau, ArcGIS Story Maps, and data storytelling means you can also "
        "create the visual outputs that make our satellite data understandable to non-technical stakeholders "
        "like conservation NGOs."
    )
    pdf.ln(2)

    # Role Description
    pdf.section_title("Your Role on ClimateVision")
    pdf.body_text(
        "You own the entire data layer - everything that happens between raw satellite imagery arriving from "
        "APIs and clean, model-ready data being passed to the ML pipeline. You are the gatekeeper of data quality."
    )
    pdf.subsection_title("Core Responsibilities")
    pdf.bullet("Build and maintain the automated satellite data ingestion pipeline (Sentinel Hub, Google Earth Engine)")
    pdf.bullet("Develop preprocessing workflows: cloud masking, atmospheric correction, image normalization, tiling")
    pdf.bullet("Create PyTorch Dataset & DataLoader classes for training and inference")
    pdf.bullet("Implement data augmentation strategies (rotation, flipping, spectral perturbations)")
    pdf.bullet("Engineer spectral features: NDVI, EVI, moisture indices from raw multispectral bands")
    pdf.bullet("Build data validation and quality checks for incoming satellite imagery")
    pdf.bullet("Manage the data/ directory structure (raw, processed, satellite)")
    pdf.bullet("Create EDA notebooks for spatial data exploration and visualization")
    pdf.ln(2)

    # Codebase Ownership
    pdf.section_title("Your Codebase Ownership")
    pdf.body_text("You are the primary owner of the following files and directories:")
    pdf.code_block(
        "src/climatevision/data/              # PRIMARY OWNER - Entire data module\n"
        "  sentinel2.py                        # Sentinel-2 downloader & preprocessor\n"
        "  landsat.py                          # Landsat data loader\n"
        "  dataset.py                          # PyTorch Dataset classes\n"
        "  preprocess.py                       # Cloud masking, normalization\n"
        "  augmentation.py                     # Data augmentation pipeline\n"
        "  __init__.py                         # Module exports\n"
        "\n"
        "src/climatevision/utils/\n"
        "  geospatial.py                       # CO-OWNER - Geospatial utilities\n"
        "  visualization.py                    # CO-OWNER - Spatial visualizations\n"
        "\n"
        "scripts/\n"
        "  setup_gee.py                        # Google Earth Engine setup\n"
        "  download_data.py                    # Automated satellite data download\n"
        "\n"
        "data/                                 # Data directory structure\n"
        "  raw/ | processed/ | satellite/\n"
        "\n"
        "notebooks/\n"
        "  02_data_exploration.ipynb            # EDA notebook"
    )
    pdf.ln(2)

    # 3-Month Timeline
    pdf.section_title("Your 3-Month Delivery Timeline")
    pdf.month_block("MONTH 1: Foundation (Weeks 1-4)", [
        ("Week 1-2: Data Ingestion", [
            "Set up Sentinel Hub API and Google Earth Engine authentication",
            "Build sentinel2.py - download, parse, and store Sentinel-2 imagery",
            "Create landsat.py - Landsat 8/9 data loader with band mapping",
            "Implement basic cloud masking using SCL (Scene Classification Layer)",
        ]),
        ("Week 3-4: PyTorch Data Pipeline", [
            "Build dataset.py - PyTorch Dataset class for satellite image tiles",
            "Implement preprocess.py - normalization, atmospheric correction, tiling (256x256)",
            "Create data validation checks (band count, resolution, CRS consistency)",
            "Write 02_data_exploration.ipynb - EDA notebook with sample visualizations",
        ]),
    ])
    pdf.month_block("MONTH 2: Advanced Features (Weeks 5-8)", [
        ("Week 5-6: Feature Engineering & Augmentation", [
            "Implement spectral index calculation: NDVI, EVI, SAVI, moisture indices",
            "Build augmentation.py using albumentations (rotation, flip, spectral noise)",
            "Add temporal compositing - median/max NDVI composites over time windows",
        ]),
        ("Week 7-8: Scale & Performance", [
            "Integrate Dask for distributed preprocessing of large image collections",
            "Optimize data loading with parallel I/O and memory-mapped files",
            "Build data caching layer for preprocessed tiles",
        ]),
    ])
    pdf.month_block("MONTH 3: Production Readiness (Weeks 9-12)", [
        ("Week 9-10: Quality & Validation", [
            "Implement data validation framework (schema checks, anomaly detection)",
            "Set up DVC (Data Version Control) for dataset tracking",
            "Create data quality reports and monitoring dashboards",
        ]),
        ("Week 11-12: Documentation & Integration", [
            "Write comprehensive docstrings and module documentation",
            "Integration testing with ML pipeline (ensure DataLoader feeds models correctly)",
            "Create data pipeline tutorial notebook for onboarding",
        ]),
    ])

    # Git Workflow
    pdf.section_title("Your Git Workflow")
    pdf.body_text("Follow this branching convention for all your work:")
    pdf.code_block(
        "# Create feature branches from develop\n"
        "git checkout develop\n"
        "git pull origin develop\n"
        "git checkout -b feature/data-sentinel2-loader\n"
        "\n"
        "# Your branch naming convention:\n"
        "feature/data-*          (new data features)\n"
        "fix/data-*              (bug fixes in data module)\n"
        "refactor/data-*         (restructuring data code)"
    )
    pdf.body_text(
        "All PRs go to the develop branch. PRs require at least 1 review from another team member. "
        "Tag @edoh-Onuh or @franchaise for data-related reviews since they consume your data outputs."
    )
    pdf.ln(3)

    # Key Collaborators
    pdf.section_title("Your Key Collaborators")
    pdf.bullet("@edoh-Onuh (ML Lead) - Your DataLoaders feed directly into their training pipeline. Coordinate on tensor shapes, normalization, and augmentation strategies.")
    pdf.bullet("@franchaise (Analytics Lead) - They need processed data for carbon estimation. Align on feature formats and metadata.")
    pdf.bullet("Olufemi Taiwo (API Lead) - Inference pipeline uses your preprocessing code. Ensure consistency between training and inference data paths.")
    pdf.bullet("@cutewizzy11 (Full-Stack) - Frontend map visualizations may need GeoJSON exports from your geospatial utils.")

    # Code Pipeline
    pdf.section_title("Your Code Pipeline")
    pdf.body_text("This is your end-to-end working pipeline from environment setup to pushing code.")

    pdf.subsection_title("Step 1: Environment Setup")
    pdf.code_block(
        "# Clone and install dependencies\n"
        "git clone https://github.com/Climate-Vision/ClimateVision.git\n"
        "cd ClimateVision\n"
        "pip install -r requirements.txt\n"
        "\n"
        "# Authenticate Google Earth Engine\n"
        "python scripts/setup_gee.py\n"
        "# Follow browser prompt to authorise your GEE service account"
    )

    pdf.subsection_title("Step 2: Ingest Satellite Data")
    pdf.code_block(
        "# Download Sentinel-2 imagery for a bounding box and date range\n"
        "python scripts/prepare_data.py \\\n"
        "  --bbox \"-60,-15,-45,5\" \\\n"
        "  --start 2023-01-01 \\\n"
        "  --end   2023-12-31 \\\n"
        "  --source sentinel2 \\\n"
        "  --output data/raw/amazon_2023\n"
        "\n"
        "# Output: GeoTIFF tiles saved to data/raw/amazon_2023/"
    )

    pdf.subsection_title("Step 3: Preprocess & Build Dataset")
    pdf.code_block(
        "# Run cloud masking, normalization, and 256x256 tiling\n"
        "python - <<'EOF'\n"
        "from climatevision.data.preprocessing import preprocess_tiles\n"
        "preprocess_tiles(\n"
        "    input_dir='data/raw/amazon_2023/',\n"
        "    output_dir='data/processed/amazon_2023/',\n"
        "    tile_size=256,\n"
        "    cloud_threshold=0.2\n"
        ")\n"
        "EOF\n"
        "\n"
        "# Validate the PyTorch dataset loads correctly\n"
        "python - <<'EOF'\n"
        "from climatevision.data.dataset import SatelliteDataset\n"
        "ds = SatelliteDataset('data/processed/amazon_2023/', split='train')\n"
        "img, mask = ds[0]\n"
        "print(f'Dataset size: {len(ds)} | Image shape: {img.shape} | Mask shape: {mask.shape}')\n"
        "EOF"
    )

    pdf.subsection_title("Step 4: Compute Spectral Indices")
    pdf.code_block(
        "# Calculate NDVI, EVI, and moisture indices from raw bands\n"
        "python - <<'EOF'\n"
        "from climatevision.utils.geospatial import compute_indices\n"
        "compute_indices(\n"
        "    tile_dir='data/processed/amazon_2023/',\n"
        "    indices=['ndvi', 'evi', 'moisture'],\n"
        "    output_dir='data/processed/amazon_2023_features/'\n"
        ")\n"
        "EOF"
    )

    pdf.subsection_title("Step 5: Commit & Push Your Work")
    pdf.code_block(
        "# Switch to your git identity\n"
        "source team_docs/switch_user.sh adeolu\n"
        "\n"
        "# Create a feature branch\n"
        "git checkout develop\n"
        "git pull origin develop\n"
        "git checkout -b feature/data-sentinel2-preprocessing\n"
        "\n"
        "# Stage your files\n"
        "git add src/climatevision/data/\n"
        "git add scripts/prepare_data.py\n"
        "\n"
        "# Commit\n"
        "git commit -m \"feat(data): add Sentinel-2 cloud masking and tile preprocessing pipeline\"\n"
        "\n"
        "# Push from your account\n"
        "git push adeolu feature/data-sentinel2-preprocessing"
    )

    pdf.output(os.path.join(OUTPUT_DIR, "Adeolu_Mary_Oshadare_Role.pdf"))
    print("Created: Adeolu_Mary_Oshadare_Role.pdf")


def create_francis_doc():
    pdf = RoleDoc("Francis Umo")
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Francis Umo", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7, "Data Science Engineer 3 - Carbon Analytics & Validation Lead", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    pdf.key_value("GitHub", "@franchaise")
    pdf.key_value("Access Level", "Maintainer")
    pdf.key_value("Reports To", "@Goldokpa (Project Owner)")
    pdf.key_value("Project Duration", "3 Months")
    pdf.ln(3)

    # How It Fits Me
    pdf.section_title("How This Role Fits You")
    pdf.body_text(
        "With 8+ years of progressive experience in data analysis and business intelligence, you bring "
        "the deepest analytical maturity on this team. While others focus on building models and pipelines, "
        "you are the person who makes sure the numbers tell the right story and that the results are trustworthy."
    )
    pdf.body_text(
        "Your expertise in Python, PostgreSQL, and SQL means you can build the carbon estimation models that "
        "require heavy data querying, aggregation, and statistical analysis. At Dataleum, you conducted data "
        "quality checks, developed dashboards to monitor financial data, and created reports that reduced fraud "
        "by 80% - that same rigour is exactly what's needed when validating whether our ML models are correctly "
        "estimating carbon loss from deforestation."
    )
    pdf.body_text(
        "Your proficiency in Tableau and Power BI is a direct match for building the impact reporting layer. "
        "ClimateVision needs to produce clear, visual reports that conservation organizations and government "
        "agencies can act on. Your data storytelling background makes you the ideal person to translate "
        "raw model outputs into actionable intelligence."
    )
    pdf.body_text(
        "Your cross-functional collaboration experience - working with IT teams, stakeholders, and bringing "
        "analytical models into production - means you understand how to bridge the gap between a data science "
        "experiment and a production metric that decision-makers rely on."
    )
    pdf.ln(2)

    # Role Description
    pdf.section_title("Your Role on ClimateVision")
    pdf.body_text(
        "You own the analytics and validation layer - everything that turns raw model predictions into "
        "meaningful environmental metrics. If the ML model says 'this pixel is deforested,' you quantify "
        "what that means in tons of carbon, hectares of forest, and dollars of environmental impact."
    )
    pdf.subsection_title("Core Responsibilities")
    pdf.bullet("Develop carbon stock estimation models (Random Forest, XGBoost regression)")
    pdf.bullet("Build biomass-to-carbon conversion pipelines using allometric equations")
    pdf.bullet("Implement uncertainty quantification (bootstrap, Monte Carlo, confidence intervals)")
    pdf.bullet("Create ground truth validation framework - compare model outputs to known data")
    pdf.bullet("Build statistical testing suite (hypothesis testing, A/B testing for model versions)")
    pdf.bullet("Design and generate impact reports (area deforested, carbon lost, trends over time)")
    pdf.bullet("Develop KPI dashboards for monitoring model performance and environmental outcomes")
    pdf.bullet("Create validation notebooks demonstrating model accuracy across regions")
    pdf.ln(2)

    # Codebase Ownership
    pdf.section_title("Your Codebase Ownership")
    pdf.body_text("You are the primary owner of the following files and directories:")
    pdf.code_block(
        "src/climatevision/analytics/          # PRIMARY OWNER - New analytics module\n"
        "  carbon.py                            # Carbon stock estimation models\n"
        "  statistics.py                        # Statistical testing & analysis\n"
        "  reporting.py                         # Impact report generation\n"
        "  validation.py                        # Ground truth validation framework\n"
        "  __init__.py                          # Module exports\n"
        "\n"
        "src/climatevision/models/\n"
        "  regression.py                        # PRIMARY OWNER - Biomass/carbon regression\n"
        "\n"
        "src/climatevision/utils/\n"
        "  metrics.py                           # CO-OWNER - Extend with carbon metrics\n"
        "\n"
        "notebooks/\n"
        "  03_carbon_analysis.ipynb             # Carbon estimation analysis\n"
        "  04_model_validation.ipynb            # Validation & benchmarking\n"
        "  05_impact_reporting.ipynb            # Reporting notebook\n"
        "\n"
        "outputs/\n"
        "  reports/                             # Generated impact reports\n"
        "  dashboards/                          # Dashboard configs"
    )
    pdf.ln(2)

    # 3-Month Timeline
    pdf.section_title("Your 3-Month Delivery Timeline")
    pdf.month_block("MONTH 1: Foundation (Weeks 1-4)", [
        ("Week 1-2: Carbon Estimation Models", [
            "Research allometric equations for biomass estimation by forest type",
            "Build carbon.py - Random Forest & XGBoost regression for biomass prediction",
            "Create feature pipeline: spectral indices -> biomass -> carbon conversion",
            "Implement metrics for regression evaluation (RMSE, MAE, R-squared)",
        ]),
        ("Week 3-4: Validation Framework", [
            "Build validation.py - compare model predictions to ground truth datasets",
            "Source and integrate reference data (Global Forest Watch, forest inventory data)",
            "Create confusion matrix, precision/recall analysis for segmentation outputs",
            "Write 04_model_validation.ipynb with baseline validation results",
        ]),
    ])
    pdf.month_block("MONTH 2: Advanced Analytics (Weeks 5-8)", [
        ("Week 5-6: Uncertainty & Statistical Testing", [
            "Implement bootstrap confidence intervals for carbon estimates",
            "Build Monte Carlo simulation for uncertainty propagation",
            "Create statistics.py - hypothesis testing, trend analysis functions",
            "Implement A/B testing framework for comparing model versions",
        ]),
        ("Week 7-8: Impact Reporting", [
            "Build reporting.py - automated report generation (PDF/HTML)",
            "Design KPI framework: hectares lost, carbon tons, trend direction",
            "Create 05_impact_reporting.ipynb - template for regional impact reports",
            "Integrate with PostgreSQL for historical metric storage",
        ]),
    ])
    pdf.month_block("MONTH 3: Production Readiness (Weeks 9-12)", [
        ("Week 9-10: Dashboard & Integration", [
            "Build dashboard data endpoints (feed metrics to frontend charts)",
            "Create time-series analysis for deforestation trend tracking",
            "Implement anomaly detection for unusual forest loss patterns",
        ]),
        ("Week 11-12: Documentation & Case Studies", [
            "Produce 3 regional case study reports (Amazon, Congo, Southeast Asia)",
            "Write comprehensive documentation for analytics module",
            "Final validation sweep across all model outputs",
            "Performance benchmarking and accuracy documentation",
        ]),
    ])

    # Git Workflow
    pdf.section_title("Your Git Workflow")
    pdf.code_block(
        "# Create feature branches from develop\n"
        "git checkout develop\n"
        "git pull origin develop\n"
        "git checkout -b feature/analytics-carbon-estimation\n"
        "\n"
        "# Your branch naming convention:\n"
        "feature/analytics-*     (new analytics features)\n"
        "fix/analytics-*         (bug fixes)\n"
        "refactor/analytics-*    (code restructuring)"
    )
    pdf.body_text(
        "All PRs go to the develop branch. PRs require at least 1 review. "
        "Tag @edoh-Onuh for reviews on model evaluation metrics, and @Oshgig for data format questions."
    )
    pdf.ln(3)

    # Key Collaborators
    pdf.section_title("Your Key Collaborators")
    pdf.bullet("@edoh-Onuh (ML Lead) - Their model predictions are your primary input. Coordinate on output formats, probability thresholds, and confidence scores.")
    pdf.bullet("@Oshgig (Data Pipeline Lead) - She provides the preprocessed data you need for carbon regression features. Align on spectral indices and metadata.")
    pdf.bullet("Olufemi Taiwo (API Lead) - Your analytics endpoints need to be exposed through the API. Coordinate on response schemas.")
    pdf.bullet("@cutewizzy11 (Full-Stack) - Frontend dashboards visualize your metrics. Provide JSON data contracts for charts.")

    # Code Pipeline
    pdf.section_title("Your Code Pipeline")
    pdf.body_text("Your pipeline starts where the ML model ends - taking prediction masks and turning them into carbon impact numbers and stakeholder reports.")

    pdf.subsection_title("Step 1: Environment Setup")
    pdf.code_block(
        "git clone https://github.com/Climate-Vision/ClimateVision.git\n"
        "cd ClimateVision\n"
        "pip install -r requirements.txt\n"
        "\n"
        "# Verify analytics dependencies\n"
        "python -c \"import xgboost, sklearn, mlflow, optuna; print('Analytics stack ready')\""
    )

    pdf.subsection_title("Step 2: Run Inference to Get Prediction Masks")
    pdf.code_block(
        "# Generate deforestation masks from a trained model\n"
        "python scripts/infer.py \\\n"
        "  --bbox \"-60,-15,-45,5\" \\\n"
        "  --date 2023-06-01 \\\n"
        "  --analysis_type deforestation \\\n"
        "  --output outputs/masks/\n"
        "\n"
        "# Output: outputs/masks/deforestation_mask.tif + confidence_scores.npy"
    )

    pdf.subsection_title("Step 3: Estimate Carbon Loss")
    pdf.code_block(
        "# Run carbon stock estimation on the prediction mask\n"
        "python - <<'EOF'\n"
        "from climatevision.analytics.carbon import estimate_carbon\n"
        "result = estimate_carbon(\n"
        "    mask_path='outputs/masks/deforestation_mask.tif',\n"
        "    region='amazon',\n"
        "    forest_type='tropical_moist'\n"
        ")\n"
        "print(f\"Deforested area: {result['hectares']:.1f} ha\")\n"
        "print(f\"Carbon lost:     {result['carbon_tonnes']:.1f} tCO2e\")\n"
        "print(f\"Confidence CI:   {result['ci_lower']:.1f} - {result['ci_upper']:.1f} tCO2e\")\n"
        "EOF"
    )

    pdf.subsection_title("Step 4: Validate Against Ground Truth")
    pdf.code_block(
        "# Compare model outputs to Global Forest Watch reference data\n"
        "python - <<'EOF'\n"
        "from climatevision.analytics.validation import validate_predictions\n"
        "metrics = validate_predictions(\n"
        "    pred_mask='outputs/masks/deforestation_mask.tif',\n"
        "    ground_truth='data/ground_truth/amazon_gfw_2023.tif'\n"
        ")\n"
        "print(f\"IoU: {metrics['iou']:.3f} | F1: {metrics['f1']:.3f} | Precision: {metrics['precision']:.3f}\")\n"
        "EOF"
    )

    pdf.subsection_title("Step 5: Generate Impact Report")
    pdf.code_block(
        "# Auto-generate a PDF/HTML impact report for stakeholders\n"
        "python - <<'EOF'\n"
        "from climatevision.analytics.reporting import generate_report\n"
        "generate_report(\n"
        "    region='amazon',\n"
        "    period='2023-Q2',\n"
        "    carbon_result=result,\n"
        "    validation_metrics=metrics,\n"
        "    output_dir='outputs/reports/'\n"
        ")\n"
        "EOF\n"
        "\n"
        "# Output: outputs/reports/amazon_2023-Q2_impact_report.pdf"
    )

    pdf.subsection_title("Step 7: Commit & Push Your Work")
    pdf.code_block(
        "# Switch to your git identity\n"
        "source team_docs/switch_user.sh francis\n"
        "\n"
        "git checkout develop && git pull origin develop\n"
        "git checkout -b feature/analytics-carbon-estimation\n"
        "\n"
        "git add src/climatevision/analytics/\n"
        "git add notebooks/03_carbon_analysis.ipynb\n"
        "git commit -m \"feat(analytics): add carbon stock estimation with confidence intervals\"\n"
        "\n"
        "git push francis feature/analytics-carbon-estimation"
    )

    pdf.output(os.path.join(OUTPUT_DIR, "Francis_Umo_Role.pdf"))
    print("Created: Francis_Umo_Role.pdf")


def create_olufemi_doc():
    pdf = RoleDoc("Olufemi Taiwo")
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Olufemi Taiwo", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7, "Data Science Engineer 4 - API & Data Quality Lead", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    pdf.key_value("GitHub", "(To be assigned)")
    pdf.key_value("Access Level", "Maintainer")
    pdf.key_value("Reports To", "@Goldokpa (Project Owner)")
    pdf.key_value("Project Duration", "3 Months")
    pdf.ln(3)

    # How It Fits Me
    pdf.section_title("How This Role Fits You")
    pdf.body_text(
        "Your current role as Reporting and Data Quality Officer at the Royal Marsden NHS Foundation Trust "
        "is the clearest signal for this assignment. Every working day you validate data flows, investigate "
        "mismatches across Epic EPR, troubleshoot system errors using SQL, and hold the line on reporting "
        "accuracy for senior clinical stakeholders. That obsessive attention to data integrity at every step "
        "from input to output is exactly what ClimateVision's API and inference pipeline need."
    )
    pdf.body_text(
        "At Fidelity Bank, you kept payment platforms reliable around the clock as an Application Support "
        "Analyst - triaging incidents, analysing root causes, and producing service reports that guided "
        "operational decisions. ClimateVision runs a similar system: satellite images arrive as requests, "
        "the API must respond correctly and quickly, and any failure needs to be caught, logged, and "
        "escalated before it reaches users. That is your wheelhouse."
    )
    pdf.body_text(
        "Your Business Intelligence work at Dataleum - building Power BI dashboards, conducting data quality "
        "checks, achieving 98% GDPR compliance - means you already understand auditability. In a climate "
        "monitoring system used by NGOs and government agencies, every prediction must be traceable, every "
        "alert explainable, and every data flow compliant. You build that confidence layer."
    )
    pdf.body_text(
        "Your ITIL 4 certification is a direct fit for incident management, change control, and problem "
        "management in production. Combined with your MSc in Data Science, you are the person who makes "
        "the API not just functional, but operationally trustworthy - with structured logging, audit trails, "
        "validated schemas, and monitoring that surfaces issues before users notice them."
    )
    pdf.ln(2)

    # Role Description
    pdf.section_title("Your Role on ClimateVision")
    pdf.body_text(
        "You own the API layer and the inference pipeline - everything between a trained model and a user "
        "receiving a validated, structured response. You ensure the system is reliable, observable, and "
        "produces outputs that are correct and auditable. You are the data quality gatekeeper for every "
        "prediction that leaves the system."
    )
    pdf.subsection_title("Core Responsibilities")
    pdf.bullet("Extend and maintain the FastAPI backend (endpoints, authentication, request validation)")
    pdf.bullet("Build Pydantic schemas for all API request/response objects - the contract for data quality")
    pdf.bullet("Implement structured logging, error handling, and audit trails throughout the inference flow")
    pdf.bullet("Build the inference validation layer - catch bad inputs, validate outputs, flag anomalies")
    pdf.bullet("Create the deforestation alert system with configurable thresholds and notification routing")
    pdf.bullet("Build API monitoring endpoints: health checks, data quality metrics, run status dashboards")
    pdf.bullet("Write SQL queries and admin endpoints for operational reporting and data audits")
    pdf.bullet("Design and document the API contract (request/response schemas, error codes, versioning)")
    pdf.ln(2)

    # Codebase Ownership
    pdf.section_title("Your Codebase Ownership")
    pdf.body_text("You are the primary owner of the following files and directories:")
    pdf.code_block(
        "src/climatevision/inference/           # PRIMARY OWNER\n"
        "  pipeline.py                          # Core inference pipeline\n"
        "  batch_processor.py                   # Batch processing with job queuing\n"
        "  postprocess.py                       # Output filtering & thresholding\n"
        "  alert_generator.py                   # Deforestation alert system\n"
        "  __init__.py\n"
        "\n"
        "src/climatevision/api/                 # PRIMARY OWNER\n"
        "  main.py                              # FastAPI application\n"
        "  auth.py                              # API key authentication\n"
        "  middleware.py                         # Request logging, CORS\n"
        "  schemas.py                           # Pydantic request/response schemas\n"
        "  __init__.py\n"
        "\n"
        "src/climatevision/db.py                # CO-OWNER - Database & audit queries\n"
        "\n"
        "run_api.sh                             # API startup script\n"
        "config.yaml                            # API & inference config sections"
    )
    pdf.ln(2)

    # 3-Month Timeline
    pdf.section_title("Your 3-Month Delivery Timeline")
    pdf.month_block("MONTH 1: Foundation (Weeks 1-4)", [
        ("Week 1-2: Schemas & Validation", [
            "Build schemas.py - Pydantic models for every API request and response object",
            "Extend pipeline.py with input validation: image shape, band count, coordinate bounds",
            "Add structured JSON logging throughout the inference flow (request ID, timestamps, errors)",
            "Implement output validation - flag predictions outside expected confidence ranges",
        ]),
        ("Week 3-4: API Hardening", [
            "Implement auth.py - API key authentication and organisation-based access control",
            "Build middleware.py - request logging, CORS, request size limits",
            "Create /api/health, /api/status, and /api/metrics endpoints for operational monitoring",
            "Write API integration tests covering validation edge cases and error responses",
        ]),
    ])
    pdf.month_block("MONTH 2: Quality & Alerts (Weeks 5-8)", [
        ("Week 5-6: Inference Quality Layer", [
            "Build postprocess.py - confidence thresholding and prediction filtering",
            "Implement anomaly detection for unusual inference outputs (flag for review)",
            "Create audit log entries for every prediction: input hash, model version, output summary",
            "Build batch_processor.py - parallel image processing with per-job status tracking",
        ]),
        ("Week 7-8: Alert System & Reporting", [
            "Build alert_generator.py - configurable deforestation threshold alerting",
            "Implement notification routing (email, webhook) for triggered alerts",
            "Write SQL reporting queries for run history, error rates, and data quality KPIs",
            "Create admin endpoints for operational dashboards: throughput, failure rates, alert volumes",
        ]),
    ])
    pdf.month_block("MONTH 3: Observability & Documentation (Weeks 9-12)", [
        ("Week 9-10: Monitoring & Data Quality Reports", [
            "Build a /api/reports endpoint returning data quality metrics over configurable time windows",
            "Implement request tracing: correlate API requests to inference runs to alerts",
            "Create a data quality dashboard feed (JSON) for the frontend to visualise pipeline health",
            "SQL-based audit trail queries: who requested what, when, and with what result",
        ]),
        ("Week 11-12: Documentation & Launch Readiness", [
            "Write the API reference: all endpoints, schemas, error codes, and usage examples",
            "Document the incident response runbook: what each error means and how to resolve it",
            "Security review: input sanitisation, SQL injection checks, API key rotation procedures",
            "Final integration testing with all team modules - validate end-to-end data flow",
        ]),
    ])

    # Git Workflow
    pdf.section_title("Your Git Workflow")
    pdf.code_block(
        "# Create feature branches from develop\n"
        "git checkout develop\n"
        "git pull origin develop\n"
        "git checkout -b feature/api-schemas\n"
        "\n"
        "# Your branch naming convention:\n"
        "feature/api-*           (API features & endpoints)\n"
        "feature/inference-*     (inference pipeline & validation)\n"
        "feature/schemas-*       (Pydantic schema changes)\n"
        "fix/api-*               (bug fixes)"
    )
    pdf.body_text(
        "All PRs go to the develop branch. Tag @cutewizzy11 for API contract reviews (he consumes your "
        "endpoints from the frontend) and @edoh-Onuh when touching inference logic that involves model outputs."
    )
    pdf.ln(3)

    # Key Collaborators
    pdf.section_title("Your Key Collaborators")
    pdf.bullet("@edoh-Onuh (ML Lead) - Their trained models are loaded by your inference pipeline. Coordinate on model format (.pth vs ONNX), input shapes, output schemas, and confidence score formats.")
    pdf.bullet("@Oshgig (Data Pipeline Lead) - Your inference input validation must match her preprocessing exactly. Align on normalization constants, expected band order, and coordinate formats.")
    pdf.bullet("@franchaise (Analytics Lead) - Their analytics endpoints are exposed through your API. Coordinate on response schemas, pagination, and data quality flags in outputs.")
    pdf.bullet("@cutewizzy11 (Full-Stack & CI/CD) - He consumes your API from the frontend and manages Docker and deployment. You two define the API contract together - endpoints, schemas, error codes.")

    # Code Pipeline
    pdf.section_title("Your Code Pipeline")
    pdf.body_text("Your pipeline covers running and validating the FastAPI server, testing all endpoints, enforcing data quality, and maintaining the inference layer.")

    pdf.subsection_title("Step 1: Environment Setup")
    pdf.code_block(
        "git clone https://github.com/Climate-Vision/ClimateVision.git\n"
        "cd ClimateVision\n"
        "pip install -r requirements.txt\n"
        "\n"
        "# Set environment variables\n"
        "cp .env.example .env\n"
        "# Edit .env: set MODEL_PATH, DB_PATH, API_KEY_SECRET"
    )

    pdf.subsection_title("Step 2: Start the API Server")
    pdf.code_block(
        "# Start FastAPI in development mode with auto-reload\n"
        "uvicorn climatevision.api.main:app \\\n"
        "  --reload \\\n"
        "  --host 0.0.0.0 \\\n"
        "  --port 8000\n"
        "\n"
        "# Interactive API docs available at:\n"
        "# http://localhost:8000/docs\n"
        "# http://localhost:8000/redoc"
    )

    pdf.subsection_title("Step 3: Test Prediction Endpoints")
    pdf.code_block(
        "# Test JSON prediction endpoint\n"
        "curl -X POST http://localhost:8000/predict/json \\\n"
        "  -H \"Content-Type: application/json\" \\\n"
        "  -d '{\n"
        "    \"bbox\": [-60, -15, -45, 5],\n"
        "    \"start_date\": \"2023-01-01\",\n"
        "    \"end_date\":   \"2023-12-31\",\n"
        "    \"analysis_type\": \"deforestation\"\n"
        "  }'\n"
        "\n"
        "# Test file-upload endpoint\n"
        "curl -X POST http://localhost:8000/predict/upload \\\n"
        "  -F \"file=@data/test/sample_tile.tif\" \\\n"
        "  -F \"analysis_type=flooding\"\n"
        "\n"
        "# Health check\n"
        "curl http://localhost:8000/health"
    )

    pdf.subsection_title("Step 4: Run Data Quality Checks")
    pdf.code_block(
        "# Validate all run records in the database meet schema requirements\n"
        "python - <<'EOF'\n"
        "from climatevision.db import get_db_connection, validate_run_schema\n"
        "conn = get_db_connection()\n"
        "issues = validate_run_schema(conn)\n"
        "if issues:\n"
        "    print(f'Data quality issues found: {len(issues)}')\n"
        "    for issue in issues:\n"
        "        print(f'  - {issue}')\n"
        "else:\n"
        "    print('All records pass quality checks')\n"
        "EOF"
    )

    pdf.subsection_title("Step 5: Register an NGO Organisation")
    pdf.code_block(
        "# Create an NGO organisation via the API\n"
        "curl -X POST http://localhost:8000/organizations \\\n"
        "  -H \"Content-Type: application/json\" \\\n"
        "  -d '{\n"
        "    \"name\": \"Amazon Conservation Trust\",\n"
        "    \"email\": \"alerts@amazonconservation.org\",\n"
        "    \"region\": \"amazon\"\n"
        "  }'\n"
        "\n"
        "# Add a regional monitoring subscription\n"
        "curl -X POST http://localhost:8000/organizations/1/subscriptions \\\n"
        "  -H \"Content-Type: application/json\" \\\n"
        "  -d '{\"bbox\": [-60,-15,-45,5], \"analysis_type\": \"deforestation\", \"alert_threshold\": 0.15}'"
    )

    pdf.subsection_title("Step 6: Commit & Push Your Work")
    pdf.code_block(
        "# Switch to your git identity\n"
        "source team_docs/switch_user.sh olufemi\n"
        "\n"
        "git checkout develop && git pull origin develop\n"
        "git checkout -b feature/api-input-validation\n"
        "\n"
        "git add src/climatevision/api/main.py\n"
        "git add src/climatevision/db.py\n"
        "git commit -m \"feat(api): add Pydantic input validation and audit logging to predict endpoints\"\n"
        "\n"
        "# Push from YOUR GitHub account (femi23)\n"
        "git push olufemi feature/api-input-validation"
    )

    pdf.output(os.path.join(OUTPUT_DIR, "Olufemi_Taiwo_Role.pdf"))
    print("Created: Olufemi_Taiwo_Role.pdf")


def create_edoh_doc():
    pdf = RoleDoc("Edoh-Onuh")
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Edoh-Onuh (John Edoh Onuh)", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7, "Data Science Engineer 1 - ML Model Development Lead", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    pdf.key_value("GitHub", "@edoh-Onuh")
    pdf.key_value("Access Level", "Maintainer")
    pdf.key_value("Reports To", "@Goldokpa (Project Owner)")
    pdf.key_value("Project Duration", "3 Months")
    pdf.ln(3)

    # How It Fits Me
    pdf.section_title("How This Role Fits You")
    pdf.body_text(
        "Your GitHub portfolio makes the case better than any job description could. You built JED Climate - "
        "a full-stack climate intelligence platform - independently. It has a FastAPI analytics engine serving "
        "a carbon calculator and climate predictor, PyTorch/TensorFlow ML services, real-time Recharts "
        "dashboards for CO2 levels, Arctic ice extent, and sea level rise, and a 14-service Docker Compose "
        "local stack. That is almost exactly what ClimateVision is. You already know this problem space."
    )
    pdf.body_text(
        "Your fintech-fraud-detection repo demonstrates the depth of ML engineering this role needs: "
        "XGBoost, Random Forest, and Neural Network ensembles with sub-100ms inference latency, SHAP/LIME "
        "explainability, concept drift detection, and a production-grade FastAPI serving layer. The same "
        "engineering discipline - fast, explainable, reliable model inference - is exactly what ClimateVision's "
        "deforestation detection pipeline requires."
    )
    pdf.body_text(
        "Your classification track record is consistent and strong: diabetes risk prediction (Scikit-learn), "
        "fraud detection (XGBoost + Neural Networks), text classification (NLP), and time series forecasting "
        "(Tesla stock). Every one of those is a direct analogue to forest vs. non-forest pixel segmentation - "
        "the core problem you will be solving here with U-Net and Siamese architectures."
    )
    pdf.body_text(
        "Your sustainable energy analysis and JED Climate's environmental dashboards show you genuinely "
        "understand the climate data domain - spectral trends, temporal signals, and what makes environmental "
        "metrics meaningful. That context matters when you are tuning a model to detect 5% forest loss "
        "in Sentinel-2 imagery at 10-metre resolution."
    )
    pdf.ln(2)

    # Role Description
    pdf.section_title("Your Role on ClimateVision")
    pdf.body_text(
        "You own all deep learning model architectures, the training pipeline, and model evaluation. "
        "Your goal is to train models that achieve high accuracy on forest segmentation and change "
        "detection, then package them cleanly for the inference pipeline. Carbon regression modelling "
        "sits with the Analytics Lead - your focus is purely classification and change detection."
    )
    pdf.subsection_title("Core Responsibilities")
    pdf.bullet("Improve and extend the U-Net architecture (Attention U-Net, residual connections, multi-scale features)")
    pdf.bullet("Train and evaluate the Siamese network for temporal bi-date change detection")
    pdf.bullet("Build a complete training pipeline: data loading, training loop, validation, checkpointing")
    pdf.bullet("Implement loss functions tuned for satellite imagery class imbalance (Focal Loss, Dice Loss)")
    pdf.bullet("Run hyperparameter optimisation using Optuna (learning rate, batch size, architecture depth)")
    pdf.bullet("Implement transfer learning from pretrained encoders (ResNet, EfficientNet backbones)")
    pdf.bullet("Build model evaluation framework: F1, IoU, precision-recall curves, confusion matrices")
    pdf.bullet("Export optimised models to ONNX for production inference speed")
    pdf.bullet("Implement experiment tracking with MLflow - log runs, metrics, and artefacts")
    pdf.ln(2)

    # Codebase Ownership
    pdf.section_title("Your Codebase Ownership")
    pdf.body_text("You are the primary owner of the following files and directories:")
    pdf.code_block(
        "src/climatevision/models/              # PRIMARY OWNER\n"
        "  unet.py                              # U-Net & Attention U-Net\n"
        "  siamese.py                           # Siamese change detection network\n"
        "  __init__.py\n"
        "  # Note: regression.py is owned by @franchaise (Analytics Lead)\n"
        "\n"
        "src/climatevision/training/            # PRIMARY OWNER - New module\n"
        "  trainer.py                           # Training loop & checkpointing\n"
        "  evaluator.py                         # Model evaluation framework\n"
        "  scheduler.py                         # Learning rate schedulers\n"
        "  callbacks.py                         # Early stopping, logging\n"
        "  __init__.py\n"
        "\n"
        "src/climatevision/utils/\n"
        "  metrics.py                           # CO-OWNER - Loss functions, metrics\n"
        "\n"
        "scripts/\n"
        "  run_training.py                      # Training pipeline script\n"
        "  train.py                             # Existing training script\n"
        "  hyperparameter_search.py             # Optuna hyperparameter search\n"
        "\n"
        "models/                                # Trained model weights\n"
        "models_pretrained/                     # Pretrained backbone weights"
    )
    pdf.ln(2)

    # 3-Month Timeline
    pdf.section_title("Your 3-Month Delivery Timeline")
    pdf.month_block("MONTH 1: Foundation (Weeks 1-4)", [
        ("Week 1-2: Training Infrastructure", [
            "Build trainer.py - complete training loop with mixed-precision, gradient accumulation",
            "Implement checkpointing (save best model, resume from checkpoint)",
            "Create evaluator.py - F1, IoU, precision, recall, confusion matrix",
            "Set up experiment tracking with MLflow - log all runs, hyperparameters, artefacts",
        ]),
        ("Week 3-4: Baseline Models", [
            "Train baseline U-Net on curated forest segmentation dataset",
            "Implement Focal Loss and Dice Loss for forest/non-forest class imbalance",
            "Run initial benchmarks: accuracy on Amazon, Congo, Southeast Asia test sets",
            "Document baseline results as the performance floor to beat",
        ]),
    ])
    pdf.month_block("MONTH 2: Advanced Models (Weeks 5-8)", [
        ("Week 5-6: Architecture Improvements", [
            "Implement Attention U-Net with skip connection attention gates",
            "Add ResNet/EfficientNet encoder backbone via transfer learning (ImageNet pretrained)",
            "Run hyperparameter search with Optuna (learning rate, batch size, depth, dropout)",
            "Train Siamese network for bi-temporal change detection",
        ]),
        ("Week 7-8: Model Optimisation", [
            "Implement model ensemble (U-Net + Attention U-Net prediction averaging)",
            "Build Monte Carlo Dropout for per-pixel uncertainty estimation",
            "Spatial cross-validation to prevent data leakage across adjacent image tiles",
            "Performance benchmarking across all model variants - pick production candidate",
        ]),
    ])
    pdf.month_block("MONTH 3: Production Models (Weeks 9-12)", [
        ("Week 9-10: Export & Versioning", [
            "Export best-performing models to ONNX format for fast production inference",
            "Implement model quantisation and pruning for latency reduction",
            "Set up model registry with versioning, metadata, and performance records",
            "Create model cards: accuracy, known limitations, training data, bias notes",
        ]),
        ("Week 11-12: Final Evaluation", [
            "Comprehensive evaluation on held-out test sets across all regions",
            "Ablation studies: measure contribution of each architectural choice",
            "Write model documentation and training reproduction guide",
            "Integration testing with Olufemi's inference pipeline - validate end-to-end",
        ]),
    ])

    # Git Workflow
    pdf.section_title("Your Git Workflow")
    pdf.code_block(
        "# Create feature branches from develop\n"
        "git checkout develop\n"
        "git pull origin develop\n"
        "git checkout -b feature/model-attention-unet\n"
        "\n"
        "# Your branch naming convention:\n"
        "feature/model-*         (new model architectures)\n"
        "feature/training-*      (training pipeline features)\n"
        "fix/model-*             (bug fixes)\n"
        "experiment/model-*      (experimental architectures)"
    )
    pdf.body_text(
        "All PRs go to the develop branch. Tag @Oshgig when your models require different data formats, "
        "@franchaise when evaluation metrics or output confidence formats change, and Olufemi Taiwo "
        "when touching model export formats or inference input shapes."
    )
    pdf.ln(3)

    # Key Collaborators
    pdf.section_title("Your Key Collaborators")
    pdf.bullet("@Oshgig (Data Pipeline Lead) - She builds the DataLoaders you train on. Coordinate on tensor shapes, normalization values, band order, and augmentation strategies.")
    pdf.bullet("@franchaise (Analytics Lead) - He owns carbon regression modelling and validates your classification outputs against ground truth. Share model confidence scores and prediction probability formats.")
    pdf.bullet("Olufemi Taiwo (API & Data Quality Lead) - He loads your trained models into the inference pipeline. Coordinate on model file format (.pth vs ONNX), expected input shapes, and output schema.")
    pdf.bullet("@cutewizzy11 (Full-Stack & CI/CD) - CI/CD pipeline runs your training scripts. Keep scripts deterministic, well-documented, and reproducible.")

    # Code Pipeline
    pdf.section_title("Your Code Pipeline")
    pdf.body_text("Your pipeline covers model architecture development, training, evaluation, and exporting production-ready checkpoints.")

    pdf.subsection_title("Step 1: Environment Setup")
    pdf.code_block(
        "git clone https://github.com/Climate-Vision/ClimateVision.git\n"
        "cd ClimateVision\n"
        "pip install -r requirements.txt\n"
        "\n"
        "# Verify PyTorch and GPU availability\n"
        "python -c \"import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')\""
    )

    pdf.subsection_title("Step 2: Verify Data Is Ready")
    pdf.code_block(
        "# Confirm @Oshgig's DataLoader feeds correctly into your model\n"
        "python - <<'EOF'\n"
        "from climatevision.data.dataset import SatelliteDataset\n"
        "from torch.utils.data import DataLoader\n"
        "ds = SatelliteDataset('data/processed/', split='train')\n"
        "loader = DataLoader(ds, batch_size=4, num_workers=2)\n"
        "imgs, masks = next(iter(loader))\n"
        "print(f'Batch shape: {imgs.shape} | Mask shape: {masks.shape}')\n"
        "# Expected: torch.Size([4, 13, 256, 256]) | torch.Size([4, 256, 256])\n"
        "EOF"
    )

    pdf.subsection_title("Step 3: Train Baseline U-Net")
    pdf.code_block(
        "# Train baseline segmentation model\n"
        "python scripts/train.py \\\n"
        "  --model unet \\\n"
        "  --analysis-type deforestation \\\n"
        "  --epochs 50 \\\n"
        "  --batch-size 16 \\\n"
        "  --lr 1e-4 \\\n"
        "  --checkpoint-dir models/ \\\n"
        "  --mlflow-tracking\n"
        "\n"
        "# Monitor training: open http://localhost:5000 (MLflow UI)\n"
        "mlflow ui --port 5000"
    )

    pdf.subsection_title("Step 4: Hyperparameter Search")
    pdf.code_block(
        "# Run Optuna search over learning rate, batch size, depth\n"
        "python scripts/hyperparameter_search.py \\\n"
        "  --model unet \\\n"
        "  --n-trials 50 \\\n"
        "  --study-name unet_deforestation_v1 \\\n"
        "  --metric val_iou\n"
        "\n"
        "# Best trial is automatically saved to models/best_hparam_unet.pth"
    )

    pdf.subsection_title("Step 5: Evaluate & Export Model")
    pdf.code_block(
        "# Full evaluation on held-out test set\n"
        "python scripts/evaluate.py \\\n"
        "  --checkpoint models/best_unet.pth \\\n"
        "  --split test \\\n"
        "  --analysis-type deforestation\n"
        "\n"
        "# Export to ONNX for fast production inference\n"
        "python scripts/export_model.py \\\n"
        "  --checkpoint models/best_unet.pth \\\n"
        "  --format onnx \\\n"
        "  --output models/unet_deforestation_v1.onnx"
    )

    pdf.subsection_title("Step 6: Commit & Push Your Work")
    pdf.code_block(
        "# Switch to your git identity\n"
        "source team_docs/switch_user.sh edoh\n"
        "\n"
        "git checkout develop && git pull origin develop\n"
        "git checkout -b feature/model-attention-unet\n"
        "\n"
        "git add src/climatevision/models/unet.py\n"
        "git add src/climatevision/training/\n"
        "git commit -m \"feat(model): add attention gates to U-Net encoder skip connections\"\n"
        "\n"
        "git push edoh feature/model-attention-unet"
    )

    pdf.output(os.path.join(OUTPUT_DIR, "Edoh_Onuh_Role.pdf"))
    print("Created: Edoh_Onuh_Role.pdf")


def create_victor_doc():
    pdf = RoleDoc("Victor Mbachu")
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Victor Mbachu", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7, "Senior Full-Stack Engineer & Infrastructure Co-Owner", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    pdf.key_value("GitHub", "@cutewizzy11")
    pdf.key_value("Access Level", "Co-Owner (Admin)")
    pdf.key_value("Reports To", "@Goldokpa (Project Owner)")
    pdf.key_value("Project Duration", "3 Months")
    pdf.ln(3)

    # How It Fits Me
    pdf.section_title("How This Role Fits You")
    pdf.body_text(
        "At Zeta Global you design and run distributed microservice systems handling over 2 million API "
        "requests daily with 99.9% uptime across multiple AWS regions - ECS Fargate clusters, RDS Aurora, "
        "SNS/SQS messaging, and blue-green CI/CD deployments provisioned via Terraform. You also serve as "
        "on-call engineer with a 15-minute average incident resolution time. That is the production "
        "engineering standard ClimateVision needs to reach, and you have already built it professionally."
    )
    pdf.body_text(
        "At RWS Global you containerised applications with Docker, deployed across dev, staging, and "
        "production environments, led a team of 3 engineers in Agile sprints, and maintained GitHub Actions "
        "CI/CD pipelines with TDD coverage. The Docker and deployment ownership on this project - "
        "previously unassigned - is a natural fit: you do this as part of your day job, not as a "
        "stretch task."
    )
    pdf.body_text(
        "Your stack breadth is the reason you can serve as repository co-owner rather than just a "
        "frontend contributor. React, Next.js, Vue, TypeScript, Node.js, PHP/Laravel, Python/Django - "
        "you can read and reason about the FastAPI backend, the PyTorch inference pipeline, and the "
        "React dashboard with equal confidence. Reviewing PRs across four data scientists requires "
        "that range. Your AWS Certified Cloud Practitioner and Professional Scrum Master certifications "
        "anchor both the infrastructure ownership and the project coordination function."
    )
    pdf.body_text(
        "Your AI integration experience - GPT-4 and Anthropic API work at RWS Global and PetMe - "
        "means you understand the ML serving layer you are wrapping with a frontend. When @edoh-Onuh "
        "exports a model and Olufemi builds the inference API, you are not reading foreign code. You "
        "have shipped production AI features before. Your two co-authored papers on agentic AI systems "
        "show that engagement runs deeper than implementation."
    )
    pdf.ln(2)

    # Role Description
    pdf.section_title("Your Role on ClimateVision")
    pdf.body_text(
        "You own the frontend application, the CI/CD infrastructure, and the Docker/deployment layer. "
        "As co-owner you are also the quality gate for all code entering the repository - the one "
        "person on the team who can review and reason about every layer of the stack."
    )
    pdf.subsection_title("Core Responsibilities - Frontend")
    pdf.bullet("Build the React/TypeScript dashboard with interactive Leaflet map for satellite analysis results")
    pdf.bullet("Create Recharts components for deforestation trends, carbon metrics, and model performance")
    pdf.bullet("Implement api.ts - the fully-typed API client for all FastAPI backend communication")
    pdf.bullet("Build the alert notification panel for real-time deforestation alerts")
    pdf.bullet("Implement responsive TailwindCSS design for desktop and tablet viewports")
    pdf.bullet("Create the deep-dive analysis page with region selector, date range picker, and model comparison")
    pdf.ln(1)

    pdf.subsection_title("Core Responsibilities - Infrastructure & CI/CD")
    pdf.bullet("Own the Dockerfile - multi-stage production build for the FastAPI + frontend application")
    pdf.bullet("Own docker-compose.yml - local development stack wiring API, database, and frontend services")
    pdf.bullet("Build and maintain GitHub Actions CI/CD pipelines: lint, type-check, test, and deploy on every PR")
    pdf.bullet("Manage production environment configuration - dev/staging/prod separation and secrets management")
    pdf.bullet("Serve as first responder for production incidents - triage, diagnose, and coordinate resolution")
    pdf.ln(1)

    pdf.subsection_title("Sprint Progress - April 2026")
    pdf.bullet("DONE: GitHub Actions CI pipeline (Python flake8 + pytest, frontend npm build)")
    pdf.bullet("DONE: Test scaffolding (tests/ directory with pytest fixtures)")
    pdf.bullet("DONE: Frontend build fixes (case-sensitive import paths)")
    pdf.bullet("DONE: Dependency fixes (removed gdal pip package, added email-validator)")
    pdf.bullet("PENDING: Frontend unit tests with Vitest + React Testing Library")
    pdf.bullet("PENDING: Auth UI - capture X-API-Key in AppContext")
    pdf.bullet("PENDING: WebSocket client for real-time run status")
    pdf.bullet("PENDING: Alert notification UI with severity filters")
    pdf.bullet("PENDING: Mask overlay on map component")
    pdf.bullet("PENDING: Docker Compose for full-stack local dev")
    pdf.ln(1)

    pdf.subsection_title("Core Responsibilities - Co-Owner")
    pdf.bullet("Review and merge pull requests from all team members (target: <24 hour turnaround)")
    pdf.bullet("Manage GitHub issues, milestones, project boards, and sprint planning")
    pdf.bullet("Enforce branch protection rules, code quality standards, and API contract consistency")
    pdf.bullet("Manage the release process: version tagging, changelog, and release notes")
    pdf.ln(2)

    # Codebase Ownership
    pdf.section_title("Your Codebase Ownership")
    pdf.body_text("You are the primary owner of the following files and directories:")
    pdf.code_block(
        "frontend/                              # PRIMARY OWNER - Entire frontend\n"
        "  src/\n"
        "    App.tsx                             # Main application shell\n"
        "    api.ts                              # Typed API client\n"
        "    main.tsx                            # Entry point\n"
        "    styles.css                          # TailwindCSS styles\n"
        "    components/                         # Component library\n"
        "      Map.tsx                           # Leaflet map\n"
        "      ResultsViewer.tsx                 # Prediction results\n"
        "      Charts.tsx                        # Recharts visualizations\n"
        "      AlertPanel.tsx                    # Alert notifications\n"
        "      Settings.tsx                      # User settings\n"
        "    pages/\n"
        "      Dashboard.tsx                     # Main dashboard\n"
        "      Analysis.tsx                      # Deep analysis view\n"
        "      History.tsx                       # Run history\n"
        "  package.json | vite.config.ts | tsconfig.json\n"
        "\n"
        "Dockerfile                             # PRIMARY OWNER - Multi-stage production build\n"
        "docker-compose.yml                     # PRIMARY OWNER - Local development stack\n"
        "\n"
        ".github/workflows/                     # PRIMARY OWNER\n"
        "  ci.yml                               # Continuous integration\n"
        "  deploy.yml                            # Deployment pipeline\n"
        "  tests.yml                            # Test automation\n"
        "\n"
        "tests/                                 # CO-OWNER (with all DS engineers)"
    )
    pdf.ln(2)

    # 3-Month Timeline
    pdf.section_title("Your 3-Month Delivery Timeline")
    pdf.month_block("MONTH 1: Foundation (Weeks 1-4)", [
        ("Week 1-2: Infrastructure & CI/CD", [
            "Write multi-stage Dockerfile for optimised API + frontend production image",
            "Build docker-compose.yml wiring FastAPI, SQLite/PostgreSQL, and frontend services locally",
            "Set up GitHub Actions CI: lint, type-check, pytest, and Vite build on every PR",
            "Create branch protection rules: require passing CI and 1 review before merging to develop",
        ]),
        ("Week 3-4: Frontend Architecture & Core Components", [
            "Configure React Router, Vite, TypeScript strict mode, TailwindCSS, ESLint, and Prettier",
            "Build Map.tsx - Leaflet map with GeoJSON overlay for deforestation masks",
            "Implement api.ts - fully-typed API client for all FastAPI endpoints",
            "Create Dashboard.tsx - main landing page with summary metrics and run status",
        ]),
    ])
    pdf.month_block("MONTH 2: Feature Development (Weeks 5-8)", [
        ("Week 5-6: Data Visualisation", [
            "Build Charts.tsx - Recharts components for deforestation trend lines, bar charts, gauges",
            "Create ResultsViewer.tsx - segmentation masks overlaid on satellite imagery",
            "Implement Analysis.tsx - region selector, date picker, model comparison view",
            "Set up Vitest and React Testing Library - component test coverage from the start",
        ]),
        ("Week 7-8: Real-Time & Interactivity", [
            "Build WebSocket integration for live prediction job status updates",
            "Create AlertPanel.tsx - real-time deforestation alert notification feed",
            "Implement History.tsx - paginated, filterable list of past analysis runs",
            "Build Settings.tsx - user preferences and API key management",
        ]),
    ])
    pdf.month_block("MONTH 3: Production Readiness (Weeks 9-12)", [
        ("Week 9-10: Deployment & Environment Config", [
            "Configure dev/staging/prod environment separation with secrets management",
            "Set up deployment pipeline to Vercel (frontend) and Docker-based backend hosting",
            "Implement health monitoring and automated alerting for production incidents",
            "Performance pass: code splitting, lazy loading, image optimisation, bundle analysis",
        ]),
        ("Week 11-12: Integration, Testing & Release", [
            "Full end-to-end integration testing against all backend API endpoints",
            "Responsive design audit for tablet and large desktop breakpoints",
            "Accessibility review: keyboard navigation and screen reader compatibility",
            "Manage v1.0 release: changelog, version tag, release notes, and deployment sign-off",
        ]),
    ])

    # Git Workflow
    pdf.section_title("Your Git Workflow")
    pdf.code_block(
        "# Create feature branches from develop\n"
        "git checkout develop\n"
        "git pull origin develop\n"
        "git checkout -b feature/frontend-leaflet-map\n"
        "\n"
        "# Your branch naming convention:\n"
        "feature/frontend-*     (frontend features)\n"
        "feature/infra-*        (Docker, CI/CD, deployment)\n"
        "feature/ci-*           (GitHub Actions changes)\n"
        "fix/frontend-*         (bug fixes)\n"
        "release/v*             (release branches)"
    )
    pdf.body_text(
        "As co-owner, you can merge directly to develop after self-review for frontend-only or infra-only "
        "changes. For changes touching shared Python code or API contracts, get a review from @Goldokpa "
        "or the relevant module owner."
    )
    pdf.ln(3)

    # Key Collaborators
    pdf.section_title("Your Key Collaborators")
    pdf.bullet("Olufemi Taiwo (API & Data Quality Lead) - He owns the FastAPI schemas, inference validation, and audit logging. You own the Docker image and deployment pipeline that runs his API. Define the API contract together: endpoint URLs, request/response shapes, auth headers, and error formats.")
    pdf.bullet("@franchaise (Analytics Lead) - His carbon metrics and KPI data feed your dashboard charts. Align on JSON data contracts, refresh intervals, and pagination formats.")
    pdf.bullet("@edoh-Onuh (ML Lead) - Model prediction outputs need to be visualised on the map. Coordinate on GeoJSON output format, confidence score rendering, and how prediction jobs report status via the API.")
    pdf.bullet("@Oshgig (Data Pipeline Lead) - Satellite imagery tile previews on the map may draw on her geospatial utilities. Align on tile formats, coordinate systems, and GeoJSON structures.")

    # Code Pipeline
    pdf.section_title("Your Code Pipeline")
    pdf.body_text("Your pipeline covers frontend development, Docker orchestration, CI/CD management, and full-stack integration testing.")

    pdf.subsection_title("Step 1: Environment Setup")
    pdf.code_block(
        "git clone https://github.com/Climate-Vision/ClimateVision.git\n"
        "cd ClimateVision\n"
        "\n"
        "# Backend dependencies\n"
        "pip install -r requirements.txt\n"
        "\n"
        "# Frontend dependencies\n"
        "cd frontend && npm install && cd .."
    )

    pdf.subsection_title("Step 2: Start Full Local Dev Stack")
    pdf.code_block(
        "# Option A: Docker Compose (full stack - recommended)\n"
        "docker-compose up --build\n"
        "# API:      http://localhost:8000\n"
        "# Frontend: http://localhost:5173\n"
        "# MLflow:   http://localhost:5000\n"
        "\n"
        "# Option B: Run services individually for faster iteration\n"
        "uvicorn climatevision.api.main:app --reload --port 8000 &\n"
        "cd frontend && npm run dev"
    )

    pdf.subsection_title("Step 3: Frontend Development Loop")
    pdf.code_block(
        "cd frontend\n"
        "\n"
        "# Run linting and type checks\n"
        "npm run lint\n"
        "npm run type-check\n"
        "\n"
        "# Run component tests\n"
        "npm run test\n"
        "\n"
        "# Build production bundle and check for errors\n"
        "npm run build\n"
        "\n"
        "# Preview production build locally\n"
        "npm run preview"
    )

    pdf.subsection_title("Step 4: Current CI/CD Configuration")
    pdf.body_text("The following .github/workflows/ci.yml is live and runs on every PR to main/develop:")
    pdf.code_block(
        "name: CI\n"
        "on:\n"
        "  push:\n"
        "    branches: [main, develop]\n"
        "  pull_request:\n"
        "    branches: [main, develop]\n"
        "\n"
        "jobs:\n"
        "  python:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "      - uses: actions/setup-python@v5\n"
        "        with: {python-version: '3.11'}\n"
        "      - run: sudo apt-get update && sudo apt-get install -y libgl1\n"
        "      - run: pip install -r requirements.txt && pip install -e .\n"
        "      - run: flake8 src/ --select=E9,F63,F7,F82\n"
        "      - run: pytest tests/ -v --tb=short\n"
        "\n"
        "  frontend:\n"
        "    runs-on: ubuntu-latest\n"
        "    defaults: {run: {working-directory: frontend}}\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "      - uses: actions/setup-node@v4\n"
        "        with: {node-version: '20', cache: 'npm'}\n"
        "      - run: npm ci\n"
        "      - run: npm run build"
    )
    pdf.ln(2)

    pdf.subsection_title("Step 5: Build & Test Docker Image")
    pdf.code_block(
        "# Build production Docker image\n"
        "docker build -t climatevision:latest .\n"
        "\n"
        "# Run container and verify it starts cleanly\n"
        "docker run -p 8000:8000 climatevision:latest\n"
        "\n"
        "# Check all services are healthy inside the container\n"
        "curl http://localhost:8000/health\n"
        "\n"
        "# Inspect image size and layers\n"
        "docker image inspect climatevision:latest | grep Size"
    )

    pdf.subsection_title("Step 6: Run Full CI Checks Locally")
    pdf.code_block(
        "# Simulate the GitHub Actions CI pipeline before pushing\n"
        "\n"
        "# 1. Python: lint and tests\n"
        "flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics\n"
        "flake8 src/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics\n"
        "pytest tests/ -v --tb=short\n"
        "\n"
        "# 2. Frontend: build\n"
        "cd frontend && npm run build\n"
        "\n"
        "# 3. Docker build succeeds\n"
        "docker-compose build"
    )

    pdf.subsection_title("Step 6: Commit & Push Your Work")
    pdf.code_block(
        "# Switch to your git identity\n"
        "source team_docs/switch_user.sh victor\n"
        "\n"
        "git checkout develop && git pull origin develop\n"
        "git checkout -b feature/frontend-leaflet-map\n"
        "\n"
        "git add frontend/src/components/Map.tsx\n"
        "git add frontend/src/api.ts\n"
        "git commit -m \"feat(frontend): add Leaflet map with GeoJSON deforestation overlay\"\n"
        "\n"
        "git push victor feature/frontend-leaflet-map\n"
        "\n"
        "# As co-owner: review and merge PRs from the team\n"
        "# gh pr review <PR_NUMBER> --approve\n"
        "# gh pr merge <PR_NUMBER> --squash"
    )

    pdf.output(os.path.join(OUTPUT_DIR, "Victor_Mbachu_Role.pdf"))
    print("Created: Victor_Mbachu_Role.pdf")


def create_godswill_doc():
    pdf = RoleDoc("Godswill Okoroafor Chukwu")
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Godswill Okoroafor Chukwu", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7, "Data Science Engineer 5 - ML Training, Experiment Tracking & Insights Lead", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    pdf.key_value("GitHub", "(To be assigned)")
    pdf.key_value("Access Level", "Maintainer")
    pdf.key_value("Reports To", "@Goldokpa (Project Owner)")
    pdf.key_value("Project Duration", "3 Months")
    pdf.ln(3)

    # How It Fits Me
    pdf.section_title("How This Role Fits You")
    pdf.body_text(
        "Your MSc in Big Data and Data Science Technology (Distinction) from Northumbria University is the "
        "academic backbone this role demands. You have not just studied machine learning - you have delivered "
        "it in production environments. At Amdari Inc., you built predictive and forecasting models that drove "
        "strategic revenue decisions, applied clustering to identify at-risk student groups, and automated "
        "reporting pipelines that cut manual processing time significantly. Every one of those deliverables "
        "maps directly onto what ClimateVision needs from its ML training and insights layer."
    )
    pdf.body_text(
        "Where @edoh-Onuh architects the deep learning models (U-Net, Siamese networks), you are the engineer "
        "who drives those models through rigorous training cycles, tracks every experiment, measures every "
        "metric, and extracts insights from the results. Your experience running classification, regression, "
        "and clustering pipelines in Python - combined with your Data Scientist role at Amdari - means you "
        "understand the full lifecycle: data in, model trained, results validated, insights delivered."
    )
    pdf.body_text(
        "Your proficiency in Power BI and Looker Studio is a strategic asset here. ClimateVision generates "
        "real predictions - deforestation percentages, ice extent loss, flood area - that conservation NGOs "
        "and research partners need presented clearly. You build the reporting layer that translates raw model "
        "outputs into KPI dashboards, trend reports, and alert summaries that non-technical stakeholders "
        "can act on. That is the last mile between a working model and measurable real-world impact."
    )
    pdf.body_text(
        "Your background in automating recurring reporting processes with Python and designing cross-functional "
        "dashboards means you also own the bridge between the ML pipeline and the business intelligence layer. "
        "With your DataCamp Associate Data Scientist certification and Full Stack Data Science qualification "
        "from 10Alytics, you bring both the theoretical depth and the applied toolkit that this role requires."
    )
    pdf.ln(2)

    # Role Description
    pdf.section_title("Your Role on ClimateVision")
    pdf.body_text(
        "You own the training analytics, experiment tracking, and insights reporting pipeline - the layer that "
        "connects raw model outputs to actionable environmental intelligence. While the ML Lead builds model "
        "architectures and the Data Pipeline Lead ingests satellite imagery, you are the engineer who runs "
        "training experiments at scale, tracks what works and why, measures model impact, and delivers "
        "structured insights to teams and stakeholders. You are the system's analytical conscience."
    )
    pdf.subsection_title("Core Responsibilities")
    pdf.bullet("Orchestrate model training runs using scripts/train.py and scripts/run_training.py with full experiment tracking via MLflow")
    pdf.bullet("Design and execute hyperparameter tuning experiments using Optuna to maximise IoU, F1, and Dice scores")
    pdf.bullet("Build and maintain the model evaluation pipeline - benchmarking across deforestation, ice melting, and flooding tasks")
    pdf.bullet("Implement clustering analysis on prediction outputs to identify regional environmental patterns and hotspots")
    pdf.bullet("Develop forecasting models to project deforestation trends, ice melt rates, and flood risk over time")
    pdf.bullet("Automate KPI reporting pipelines that summarise model performance and environmental metrics for NGO stakeholders")
    pdf.bullet("Design and maintain Power BI / Looker Studio dashboards tracking training progress, model accuracy, and climate impact")
    pdf.bullet("Create data quality reports that validate training datasets and flag anomalies before they reach the model")
    pdf.bullet("Produce regional impact analysis notebooks showing before/after environmental change metrics")
    pdf.bullet("Feed structured insight data to the API layer and React dashboard for live reporting")
    pdf.ln(2)

    # Codebase Ownership
    pdf.section_title("Your Codebase Ownership")
    pdf.body_text("You are the primary owner of the following files and directories:")
    pdf.code_block(
        "scripts/                              # PRIMARY OWNER - Training & evaluation scripts\n"
        "  train.py                            # Model training entry point\n"
        "  run_training.py                     # Training orchestration & scheduling\n"
        "  evaluate.py                         # Model evaluation & benchmarking\n"
        "  infer.py                            # Single inference runner\n"
        "\n"
        "src/climatevision/training/\n"
        "  trainer.py                          # CO-OWNER - Training loop, EMA, mixed precision\n"
        "  losses.py                           # CO-OWNER - Focal Loss, Dice Loss tuning\n"
        "\n"
        "src/climatevision/utils/\n"
        "  metrics.py                          # CO-OWNER - IoU, F1, Dice, recall tracking\n"
        "  visualization.py                    # CO-OWNER - Training curve & result plots\n"
        "\n"
        "notebooks/\n"
        "  06_training_analysis.ipynb          # Experiment tracking & training insights\n"
        "  07_model_benchmarking.ipynb         # Cross-task model performance comparison\n"
        "  08_regional_insights.ipynb          # Clustering & trend analysis by region\n"
        "\n"
        "outputs/\n"
        "  reports/training/                   # Training run reports\n"
        "  dashboards/kpi/                     # KPI dashboard configs\n"
        "\n"
        "logs/                                 # Training logs & MLflow run artifacts\n"
        "models/                               # Model checkpoints (coordinate with ML Lead)"
    )
    pdf.ln(2)

    # Key Impact Areas
    pdf.section_title("Your High-Impact Contributions")
    pdf.body_text(
        "Your work directly determines whether ClimateVision's models are as accurate as possible and whether "
        "their outputs are trusted by the organisations that rely on them. Three areas define your impact:"
    )
    pdf.subsection_title("1. Experiment-Driven Model Improvement")
    pdf.body_text(
        "Every training run you log is a data point. By systematically tracking learning rate schedules, "
        "augmentation strategies, loss function weights, and batch sizes via MLflow and Optuna, you will "
        "build the evidence base that drives model accuracy from baseline to production-grade. Your tuning "
        "work is the difference between a model that detects 65% of deforestation events and one that "
        "detects 85%."
    )
    pdf.subsection_title("2. Regional Clustering & Trend Forecasting")
    pdf.body_text(
        "Your clustering expertise turns raw pixel predictions into geographic intelligence. By grouping "
        "regions with similar deforestation trajectories or flood risk patterns, you reveal insights that "
        "no single prediction run can show. Paired with time-series forecasting models, you can project "
        "where the next environmental crisis is developing before it becomes catastrophic - giving NGO "
        "partners the lead time they need to act."
    )
    pdf.subsection_title("3. Stakeholder-Ready Reporting")
    pdf.body_text(
        "Raw model metrics mean nothing to a conservation officer or a policy researcher. Your Power BI "
        "and automated Python reporting pipelines convert IoU scores and segmentation masks into carbon "
        "loss estimates, hectare counts, and trend alerts that stakeholders can put in a board report. "
        "This is the last mile of impact - and you own it."
    )
    pdf.ln(2)

    # 3-Month Timeline
    pdf.section_title("Your 3-Month Delivery Timeline")
    pdf.month_block("MONTH 1: Foundation (Weeks 1-4)", [
        ("Week 1-2: Training Infrastructure & Experiment Tracking", [
            "Set up MLflow tracking server and connect to scripts/train.py",
            "Instrument trainer.py to log all hyperparameters, metrics, and artifacts per run",
            "Run baseline training experiments for deforestation, ice melting, and flooding tasks",
            "Document baseline IoU, F1, and Dice scores per analysis type",
        ]),
        ("Week 3-4: Evaluation Pipeline", [
            "Build scripts/evaluate.py - full evaluation suite with per-class metrics",
            "Extend metrics.py with precision-recall curves and confusion matrix exports",
            "Create 07_model_benchmarking.ipynb - cross-task performance comparison",
            "Identify top 3 weaknesses in baseline models and propose tuning strategies",
        ]),
    ])
    pdf.month_block("MONTH 2: Optimisation & Insights (Weeks 5-8)", [
        ("Week 5-6: Hyperparameter Tuning", [
            "Set up Optuna study for learning rate, batch size, loss weights, and augmentation",
            "Run tuning experiments targeting IoU improvement of at least 10% over baseline",
            "Log all trials in MLflow with full reproducibility (seed, config, checkpoint)",
            "Implement best-config automatic checkpoint promotion pipeline",
        ]),
        ("Week 7-8: Clustering & Trend Forecasting", [
            "Build regional clustering pipeline using K-Means / DBSCAN on prediction outputs",
            "Develop time-series forecasting models for deforestation and ice melt trends",
            "Create 08_regional_insights.ipynb - hotspot identification and trend projections",
            "Generate first set of regional environmental trend reports",
        ]),
    ])
    pdf.month_block("MONTH 3: Reporting & Production Readiness (Weeks 9-12)", [
        ("Week 9-10: KPI Dashboard & Automated Reporting", [
            "Build automated Python reporting pipeline - weekly model performance summaries",
            "Design Power BI / Looker Studio KPI dashboard (accuracy trends, alert counts, coverage)",
            "Expose dashboard data via API endpoints coordinated with Olufemi",
            "Automate NGO-facing impact reports: area affected, confidence scores, trend direction",
        ]),
        ("Week 11-12: Documentation & Final Benchmarks", [
            "Write 06_training_analysis.ipynb - full experiment history and lessons learned",
            "Produce final benchmark report comparing all model versions across 3 months",
            "Document all MLflow experiments, best checkpoints, and recommended configs",
            "Deliver 3 regional case study insight reports to the team for stakeholder use",
        ]),
    ])

    # Git Workflow
    pdf.section_title("Your Git Workflow")
    pdf.body_text("Follow this branching convention for all your work:")
    pdf.code_block(
        "# Create feature branches from develop\n"
        "git checkout develop\n"
        "git pull origin develop\n"
        "git checkout -b feature/training-mlflow-setup\n"
        "\n"
        "# Your branch naming convention:\n"
        "feature/training-*      (training pipeline features)\n"
        "feature/insights-*      (reporting and analytics features)\n"
        "fix/training-*          (bug fixes in training scripts)\n"
        "experiment/tuning-*     (hyperparameter experiment branches)"
    )
    pdf.body_text(
        "All PRs go to the develop branch. PRs require at least 1 review. "
        "Tag @edoh-Onuh for model architecture questions and @franchaise for analytics overlap reviews. "
        "Always attach MLflow run IDs in PRs that change training logic so reviewers can verify metrics."
    )
    pdf.ln(3)

    # Key Collaborators
    pdf.section_title("Your Key Collaborators")
    pdf.bullet("@edoh-Onuh (ML Model Development Lead) - You run the training experiments on their model architectures. Coordinate on loss function choices, training hyperparameters, and checkpoint formats. Their architecture decisions constrain your tuning search space.")
    pdf.bullet("@Oshgig (Data Pipeline Lead) - Your training runs consume her PyTorch DataLoaders. Align on tensor shapes, normalization ranges, augmentation strategies, and the data split structure (train/val/test).")
    pdf.bullet("@franchaise (Carbon Analytics Lead) - Your model evaluation outputs are the input to their carbon estimation and validation work. Provide segmentation mask formats, confidence scores, and per-class metrics in agreed schemas.")
    pdf.bullet("Olufemi Taiwo (API & Data Quality Lead) - Your KPI reporting data needs to be surfaced via API endpoints. Coordinate on response formats, refresh cycles, and how training run metadata is exposed to the dashboard.")
    pdf.bullet("Victor Mbachu (Full-Stack & Infrastructure) - Your dashboard configs and reporting outputs feed the React frontend visualisations. Align on JSON contracts for time-series charts, gauge metrics, and alert summaries.")

    # Code Pipeline
    pdf.section_title("Your Code Pipeline")
    pdf.body_text("Your pipeline covers experiment tracking setup, running and tuning training jobs, evaluating model performance, and generating insight reports for stakeholders.")

    pdf.subsection_title("Step 1: Environment Setup")
    pdf.code_block(
        "git clone https://github.com/Climate-Vision/ClimateVision.git\n"
        "cd ClimateVision\n"
        "pip install -r requirements.txt\n"
        "\n"
        "# Verify ML and analytics stack\n"
        "python -c \"import torch, mlflow, optuna, sklearn; print('ML stack ready')\"\n"
        "\n"
        "# Start MLflow tracking server\n"
        "mlflow server --host 0.0.0.0 --port 5000 &\n"
        "# Dashboard: http://localhost:5000"
    )

    pdf.subsection_title("Step 2: Run a Training Experiment")
    pdf.code_block(
        "# Run a tracked training job\n"
        "python scripts/run_training.py \\\n"
        "  --config config/deforestation.yaml \\\n"
        "  --mlflow-tracking \\\n"
        "  --experiment-name deforestation_v1\n"
        "\n"
        "# All metrics, params, and checkpoints auto-logged to MLflow\n"
        "# View results: http://localhost:5000/#/experiments"
    )

    pdf.subsection_title("Step 3: Hyperparameter Tuning with Optuna")
    pdf.code_block(
        "# Launch an Optuna study to find the best training config\n"
        "python - <<'EOF'\n"
        "import optuna, mlflow\n"
        "from climatevision.training.trainer import train_with_config\n"
        "\n"
        "def objective(trial):\n"
        "    config = {\n"
        "        'lr':         trial.suggest_float('lr', 1e-5, 1e-3, log=True),\n"
        "        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),\n"
        "        'dropout':    trial.suggest_float('dropout', 0.1, 0.5),\n"
        "    }\n"
        "    return train_with_config(config, metric='val_iou')\n"
        "\n"
        "study = optuna.create_study(direction='maximize', study_name='unet_deforestation')\n"
        "study.optimize(objective, n_trials=50)\n"
        "print(f'Best IoU: {study.best_value:.4f}')\n"
        "print(f'Best params: {study.best_params}')\n"
        "EOF"
    )

    pdf.subsection_title("Step 4: Evaluate & Benchmark Models")
    pdf.code_block(
        "# Evaluate best checkpoint across all analysis types\n"
        "python scripts/evaluate.py \\\n"
        "  --checkpoint models/best_unet.pth \\\n"
        "  --split test \\\n"
        "  --analysis-type deforestation \\\n"
        "  --export-metrics outputs/reports/training/deforestation_eval.json\n"
        "\n"
        "# Compare all model versions logged in MLflow\n"
        "python - <<'EOF'\n"
        "import mlflow\n"
        "runs = mlflow.search_runs(experiment_names=['deforestation_v1'],\n"
        "                          order_by=['metrics.val_iou DESC'])\n"
        "print(runs[['run_id','metrics.val_iou','params.lr','params.batch_size']].head(10))\n"
        "EOF"
    )

    pdf.subsection_title("Step 5: Generate Stakeholder KPI Report")
    pdf.code_block(
        "# Run clustering on prediction outputs to find regional hotspots\n"
        "python - <<'EOF'\n"
        "from sklearn.cluster import KMeans\n"
        "import numpy as np, json\n"
        "predictions = np.load('outputs/masks/deforestation_confidence.npy')\n"
        "kmeans = KMeans(n_clusters=5, random_state=42).fit(predictions.reshape(-1, 1))\n"
        "hotspot_regions = np.where(kmeans.labels_ == kmeans.cluster_centers_.argmax())[0]\n"
        "print(f'High-risk tiles identified: {len(hotspot_regions)}')\n"
        "EOF\n"
        "\n"
        "# Auto-generate weekly KPI summary report\n"
        "python - <<'EOF'\n"
        "from climatevision.analytics.reporting import generate_kpi_report\n"
        "generate_kpi_report(\n"
        "    metrics_dir='outputs/reports/training/',\n"
        "    period='2024-W12',\n"
        "    output='outputs/dashboards/kpi/weekly_summary.pdf'\n"
        ")\n"
        "EOF"
    )

    pdf.subsection_title("Step 6: Commit & Push Your Work")
    pdf.code_block(
        "# Switch to your git identity\n"
        "source team_docs/switch_user.sh godswill\n"
        "\n"
        "git checkout develop && git pull origin develop\n"
        "git checkout -b feature/training-mlflow-setup\n"
        "\n"
        "git add scripts/run_training.py\n"
        "git add scripts/evaluate.py\n"
        "git add notebooks/06_training_analysis.ipynb\n"
        "git commit -m \"feat(training): add MLflow experiment tracking and Optuna hyperparameter search\"\n"
        "\n"
        "git push godswill feature/training-mlflow-setup"
    )

    pdf.output(os.path.join(OUTPUT_DIR, "Godswill_Chukwu_Role.pdf"))
    print("Created: Godswill_Chukwu_Role.pdf")


def create_paul_doc():
    pdf = RoleDoc("Paul (cutewizzy11)")
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Paul", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7, "Frontend Developer - React Dashboard & UI Lead", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    pdf.key_value("GitHub", "@cutewizzy11")
    pdf.key_value("Access Level", "Maintainer")
    pdf.key_value("Reports To", "@Goldokpa (Project Owner)")
    pdf.key_value("Project Duration", "3 Months")
    pdf.ln(3)

    # How It Fits
    pdf.section_title("How This Role Fits You")
    pdf.body_text(
        "Your GitHub portfolio shows a developer who is comfortable across the full stack but has a clear "
        "strength in TypeScript and JavaScript-driven interfaces. nova-agent, Data-management-Koinonia, "
        "and anyebe-web-craft are all TypeScript projects - the same language ClimateVision's frontend is "
        "built in. Your react-projects and ecommerce-app repositories show hands-on React experience, and "
        "your Heart-Attack-Risk-Predictor on Streamlit shows you can bridge data science outputs and "
        "interactive user interfaces - exactly the challenge you face here."
    )
    pdf.body_text(
        "ClimateVision's dashboard already has a working foundation: React 18, TypeScript strict mode, "
        "Vite, TailwindCSS, React Router, Recharts, and a fully-typed API client. Your job is not to "
        "start from scratch - it is to take this functional base and build the components, pages, and "
        "interactions that turn it into a polished, production-ready environmental monitoring dashboard "
        "that NGOs and researchers can actually use."
    )
    pdf.body_text(
        "Your experience with data management interfaces (Koinonia church app) and e-commerce UIs means "
        "you understand how to build interfaces where users interact with structured data - filtering, "
        "searching, viewing records, managing subscriptions. That skill maps directly onto ClimateVision's "
        "run history browser, NGO subscription manager, and alert tracking panel. You have shipped this "
        "category of UI before."
    )
    pdf.ln(2)

    # Role Description
    pdf.section_title("Your Role on ClimateVision")
    pdf.body_text(
        "You own the React dashboard - every pixel the end user sees. The backend API is built, the "
        "data models are defined, and the component library has a strong foundation. Your mission is "
        "to complete the user-facing layer: build missing pages, wire components to live API data, "
        "implement real-time updates, and ensure the interface is responsive, accessible, and fast. "
        "You are the engineer who makes ClimateVision feel like a real product."
    )
    pdf.subsection_title("Core Responsibilities")
    pdf.bullet("Build and complete all dashboard pages: Dashboard home, NGO Management, Alerts, and deep-dive Analysis views")
    pdf.bullet("Wire all components to live API data using the existing api.ts client - replace mock/static data throughout")
    pdf.bullet("Implement real-time run status updates using polling (useRunPolling hook) and WebSocket for live job tracking")
    pdf.bullet("Build the NGO management page - organisation registration, subscription setup, alert acknowledgment")
    pdf.bullet("Implement the Alerts page - filterable, paginated alert feed with severity badges and map drill-down")
    pdf.bullet("Extend the Map components - overlay segmentation masks on the map after prediction completes")
    pdf.bullet("Add component-level tests using Vitest and React Testing Library")
    pdf.bullet("Ensure full responsive design for tablet and desktop breakpoints using TailwindCSS")
    pdf.bullet("Implement accessibility: keyboard navigation, screen reader labels, focus management")
    pdf.bullet("Performance: code splitting, lazy loading pages, skeleton loading states already in the UI library")
    pdf.ln(2)

    # Codebase Ownership
    pdf.section_title("Your Codebase Ownership")
    pdf.body_text("You are the primary owner of the entire frontend directory:")
    pdf.code_block(
        "frontend/src/                          # PRIMARY OWNER - Full frontend\n"
        "\n"
        "  pages/                               # PRIMARY OWNER - All page components\n"
        "    NewAnalysis.tsx                    # Exists - extend with live map result overlay\n"
        "    Upload.tsx                         # Exists - connect to /predict/upload endpoint\n"
        "    RunHistory.tsx                     # Exists - add filters, pagination, search\n"
        "    Analytics.tsx                      # Exists - connect live data, add date picker\n"
        "    Settings.tsx                       # Exists - wire to API key and config endpoints\n"
        "    Dashboard.tsx                      # BUILD - Home page KPI summary\n"
        "    NGOManagement.tsx                  # BUILD - Org registration + subscriptions\n"
        "    Alerts.tsx                         # BUILD - Alert feed with severity filters\n"
        "\n"
        "  components/                          # PRIMARY OWNER - All UI components\n"
        "    charts/                            # Extend existing Recharts components\n"
        "    Map/                               # Extend - add mask overlay on results\n"
        "    ngo/                               # Complete - wire AlertsPanel, SubscriptionManager\n"
        "    results/                           # Complete - wire ResultsPanel to live predictions\n"
        "    runs/                              # Extend RunCard with status polling\n"
        "    ui/                                # Extend UI library as needed\n"
        "\n"
        "  api.ts                               # CO-OWNER - Add any missing endpoint calls\n"
        "  types.ts                             # CO-OWNER - Add frontend-specific types\n"
        "  contexts/                            # CO-OWNER - AppContext, ToastContext\n"
        "  hooks/                               # PRIMARY OWNER - useGeocoding, useRunPolling\n"
        "\n"
        "  tests/                               # PRIMARY OWNER - Component tests (to be created)\n"
        "    components/\n"
        "    pages/"
    )
    pdf.ln(2)

    # 3-Month Timeline
    pdf.section_title("Your 3-Month Delivery Timeline")
    pdf.month_block("MONTH 1: Foundation & Live Data (Weeks 1-4)", [
        ("Week 1-2: Setup & API Wiring", [
            "Clone repo, install deps, run dev server - verify all pages render",
            "Run the FastAPI backend locally and confirm api.ts endpoints connect",
            "Wire RunHistory page to live /runs API data - replace any static data",
            "Wire Analytics page to live run metrics - confirm charts render with real data",
            "Add loading skeletons (SkeletonCard already exists) to all data-fetching pages",
        ]),
        ("Week 3-4: Dashboard Home & Settings", [
            "Build Dashboard.tsx - KPI summary cards: total runs, alerts, analysis breakdown",
            "Add Dashboard as the new root route (/) and move NewAnalysis to /new-analysis",
            "Wire Settings.tsx to API config endpoints - API base URL, analysis preferences",
            "Implement Toast notifications for success/error states across all forms",
        ]),
    ])
    pdf.month_block("MONTH 2: NGO Features & Real-Time (Weeks 5-8)", [
        ("Week 5-6: NGO Management Page", [
            "Build NGOManagement.tsx - list registered organisations from /organizations endpoint",
            "Implement organisation registration form with validation",
            "Build SubscriptionManager UI - region bbox picker + analysis type + threshold",
            "Wire to POST /organizations and POST /organizations/{id}/subscriptions endpoints",
        ]),
        ("Week 7-8: Alerts & Real-Time Updates", [
            "Build Alerts.tsx - paginated alert feed filtered by severity and analysis type",
            "Implement alert acknowledgment button wired to PATCH /organizations/{id}/alerts/{id}",
            "Extend useRunPolling hook to poll job status and update UI when predictions complete",
            "Add live segmentation mask overlay on RegionMap after a prediction run finishes",
        ]),
    ])
    pdf.month_block("MONTH 3: Polish & Production (Weeks 9-12)", [
        ("Week 9-10: Testing & Accessibility", [
            "Set up Vitest and React Testing Library - write tests for all page components",
            "Test all API integration points with mocked responses",
            "Accessibility audit: add aria-labels, keyboard nav, focus rings across all pages",
            "Responsive design audit - tablet (768px) and large desktop (1440px) breakpoints",
        ]),
        ("Week 11-12: Performance & Final Integration", [
            "Implement React.lazy() and Suspense for all page-level code splitting",
            "Bundle analysis with vite-bundle-visualizer - eliminate unused dependencies",
            "Full end-to-end test: bbox input -> prediction job -> live status -> result on map",
            "Final UI polish pass: spacing, typography, colour consistency across all pages",
        ]),
    ])

    # Code Pipeline
    pdf.section_title("Your Code Pipeline")
    pdf.body_text("Your daily pipeline as frontend developer - from clone to a live feature pushed to GitHub.")

    pdf.subsection_title("Step 1: Setup")
    pdf.code_block(
        "git clone https://github.com/Climate-Vision/ClimateVision.git\n"
        "cd ClimateVision/frontend\n"
        "npm install\n"
        "\n"
        "# Start the backend API (needed for live data)\n"
        "cd .. && uvicorn climatevision.api.main:app --reload --port 8000 &\n"
        "\n"
        "# Start the frontend dev server\n"
        "cd frontend && npm run dev\n"
        "# App running at: http://localhost:5173"
    )

    pdf.subsection_title("Step 2: Build a New Page or Component")
    pdf.code_block(
        "# Example: building the Dashboard home page\n"
        "touch src/pages/Dashboard.tsx\n"
        "\n"
        "# Import existing UI primitives - don't rebuild what exists\n"
        "# Available: Card, Badge, StatusBadge, SkeletonCard, ProgressBar,\n"
        "#            Tooltip, EmptyState, ErrorBoundary, AnalysisTypeSelector\n"
        "\n"
        "# Import charts - already built with Recharts\n"
        "# Available: TimeSeriesChart, BarChart, GaugeChart\n"
        "\n"
        "# Import API functions from api.ts\n"
        "# import { listRuns, listOrganizations, listAlerts } from '../api'"
    )

    pdf.subsection_title("Step 3: Connect to Live API Data")
    pdf.code_block(
        "# Example: fetching live runs in a component\n"
        "import { useEffect, useState } from 'react'\n"
        "import { listRuns } from '../api'\n"
        "import type { Run } from '../api'\n"
        "\n"
        "const [runs, setRuns] = useState<Run[]>([])\n"
        "const [loading, setLoading] = useState(true)\n"
        "\n"
        "useEffect(() => {\n"
        "  listRuns().then(data => {\n"
        "    setRuns(data)\n"
        "    setLoading(false)\n"
        "  })\n"
        "}, [])\n"
        "\n"
        "# Use SkeletonCard while loading\n"
        "if (loading) return <SkeletonCard />"
    )

    pdf.subsection_title("Step 4: Run Quality Checks")
    pdf.code_block(
        "# From the frontend/ directory:\n"
        "\n"
        "# TypeScript type check - zero errors before pushing\n"
        "npm run type-check\n"
        "\n"
        "# Lint check\n"
        "npm run lint\n"
        "\n"
        "# Run component tests\n"
        "npm run test\n"
        "\n"
        "# Production build - must succeed before any PR\n"
        "npm run build"
    )

    pdf.subsection_title("Step 5: Commit & Push Your Work")
    pdf.code_block(
        "# Switch to your git identity\n"
        "source team_docs/switch_user.sh paul\n"
        "\n"
        "git checkout develop && git pull origin develop\n"
        "git checkout -b feature/frontend-dashboard-home\n"
        "\n"
        "# Stage only frontend files\n"
        "git add frontend/src/pages/Dashboard.tsx\n"
        "git add frontend/src/main.tsx\n"
        "\n"
        "git commit -m \"feat(frontend): add Dashboard home page with KPI summary cards\"\n"
        "\n"
        "# Push from your GitHub account\n"
        "git push paul feature/frontend-dashboard-home\n"
        "\n"
        "# Branch naming convention:\n"
        "# feature/frontend-*    new UI features\n"
        "# fix/frontend-*        bug fixes\n"
        "# refactor/frontend-*   component refactoring"
    )

    pdf.section_title("Your Key Collaborators")
    pdf.bullet("Olufemi Taiwo (femi23) - He owns the FastAPI backend your api.ts calls. Any new endpoint you need, request it from him. Coordinate on response shapes, pagination, and error formats.")
    pdf.bullet("@Goldokpa (Project Owner) - He built the original api.ts and App shell. He is your first point of contact for architecture questions and has context on every frontend design decision.")
    pdf.bullet("@franchaise (Analytics Lead) - His carbon metrics and KPI data feed your Analytics and Dashboard pages. Agree on the JSON structure for chart data with him.")
    pdf.bullet("Victor Mbachu (@cutewizzy11 in other refs) - If Docker or CI/CD issues block your local dev, coordinate with the infrastructure owner.")
    pdf.bullet("@edoh-Onuh (ML Lead) - Model prediction outputs appear as map overlays in your UI. Coordinate on the GeoJSON mask format and confidence score schema so your map component renders them correctly.")

    pdf.output(os.path.join(OUTPUT_DIR, "Paul_cutewizzy11_Role.pdf"))
    print("Created: Paul_cutewizzy11_Role.pdf")


def create_gold_doc():
    pdf = RoleDoc("Gold Okpa")
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Gold Okpa", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7, "Project Owner & Lead Architect - ClimateVision", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    pdf.key_value("GitHub", "@Goldokpa")
    pdf.key_value("Access Level", "Owner (Admin)")
    pdf.key_value("Email", "okpagold@gmail.com")
    pdf.key_value("Project Duration", "Ongoing")
    pdf.ln(3)

    # Role Overview
    pdf.section_title("Your Role on ClimateVision")
    pdf.body_text(
        "You built ClimateVision from the ground up. Every foundational layer of this system - the React "
        "frontend and API client, the Google Earth Engine integration with service account auth and synthetic "
        "NDVI fallback, the data pipeline scripts, the training and evaluation infrastructure, the Colab "
        "training notebook, and the overall architecture - was shipped by you. You are not just the project "
        "owner in title. You are the technical architect, the integration lead, and the person who knows "
        "every module of this codebase at a deep level."
    )
    pdf.body_text(
        "As the team scales, your role shifts from building everything yourself to orchestrating six "
        "specialist engineers - setting the architectural direction, reviewing and merging their code, "
        "maintaining the integrity of the overall system, and ensuring every module fits together cleanly. "
        "You are the final authority on what goes into the main branch and what ships to users."
    )
    pdf.ln(2)

    pdf.subsection_title("Core Responsibilities")
    pdf.bullet("Own the overall system architecture and make final decisions on design patterns, module boundaries, and API contracts")
    pdf.bullet("Review and merge all pull requests into the develop and main branches")
    pdf.bullet("Maintain config.yaml - the single source of truth for all model, data, and API configuration")
    pdf.bullet("Own the Google Earth Engine integration and satellite data orchestration at the system level")
    pdf.bullet("Manage GitHub repository: branch protection rules, secrets, environment variables, and access permissions")
    pdf.bullet("Coordinate sprint planning, milestone tracking, and cross-team dependency resolution")
    pdf.bullet("Own the release process: version tagging, changelog, and production deployment sign-off")
    pdf.bullet("Onboard new team members and ensure every engineer has the access and context they need")
    pdf.bullet("Make final calls on model selection, analysis type prioritisation, and stakeholder deliverables")
    pdf.ln(2)

    # Codebase Ownership
    pdf.section_title("Your Codebase Ownership")
    pdf.body_text("As project owner you have authority over the full codebase. Your primary ownership areas are:")
    pdf.code_block(
        "config.yaml                            # PRIMARY OWNER - All system configuration\n"
        ".env / .env.example                    # PRIMARY OWNER - Environment secrets template\n"
        "setup.py / requirements.txt            # PRIMARY OWNER - Package definition\n"
        "\n"
        "src/climatevision/                     # ARCHITECT - Full codebase authority\n"
        "  api/main.py                          # Co-owner with Olufemi - original author\n"
        "  analysis/                            # Original author - analysis framework\n"
        "  config.py                            # PRIMARY OWNER - Config management\n"
        "  db.py                                # PRIMARY OWNER - Database schema\n"
        "\n"
        "scripts/                               # ORIGINAL AUTHOR - All pipeline scripts\n"
        "  prepare_data.py                      # GEE data pipeline (you built this)\n"
        "  setup_gee.py                         # GEE service account auth\n"
        "  train.py | evaluate.py | infer.py    # Training & inference scripts\n"
        "  export_model.py                      # ONNX export\n"
        "\n"
        "frontend/                              # ORIGINAL AUTHOR - App shell & API client\n"
        "  src/App.tsx                          # Main application\n"
        "  src/api.ts                           # API client (you wrote this)\n"
        "\n"
        "notebooks/                             # ORIGINAL AUTHOR\n"
        "  train_on_colab.ipynb                 # Colab training notebook\n"
        "\n"
        ".github/                               # PRIMARY OWNER - CI/CD and repo rules\n"
        "README.md / CONTRIBUTING.md            # PRIMARY OWNER - Public documentation"
    )
    pdf.ln(2)

    # 3-Month Plan
    pdf.section_title("Your 3-Month Orchestration Plan")
    pdf.month_block("MONTH 1: Team Integration (Weeks 1-4)", [
        ("Week 1-2: Onboarding & Access", [
            "Grant all 6 engineers Maintainer access on GitHub",
            "Set up branch protection: require passing CI + 1 review on develop",
            "Create GitHub project board with milestones mapped to each engineer's 3-month timeline",
            "Distribute and walk through each team member's role document",
            "Verify all engineers can clone the repo, install dependencies, and run the API locally",
        ]),
        ("Week 3-4: Architecture Alignment", [
            "Hold kickoff session: walkthrough of config.yaml, module boundaries, and API contracts",
            "Define and document tensor shapes, data formats, and model output schemas",
            "Review and merge first PRs from each team member - establish code review rhythm",
            "Set up MLflow server on shared infrastructure for experiment tracking",
        ]),
    ])
    pdf.month_block("MONTH 2: Integration & Quality (Weeks 5-8)", [
        ("Week 5-6: Cross-Module Integration", [
            "Integration test: Adeolu's DataLoader -> Edoh's model -> Olufemi's inference API",
            "Integration test: Olufemi's API output -> Francis' carbon estimation -> Victor's dashboard",
            "Resolve any data contract mismatches between modules",
            "Set up automated integration test suite in GitHub Actions",
        ]),
        ("Week 7-8: Architecture Reviews", [
            "Review all module implementations against original architecture design",
            "Identify and resolve any technical debt or design drift before it compounds",
            "Run end-to-end test: satellite bbox input -> dashboard output for all 3 analysis types",
            "Performance profiling: measure API latency and model inference time",
        ]),
    ])
    pdf.month_block("MONTH 3: Production & Release (Weeks 9-12)", [
        ("Week 9-10: Production Hardening", [
            "Review all security configurations: API keys, CORS, input validation, secrets management",
            "Final review of Docker and CI/CD pipeline with Victor",
            "Load test the API endpoints - verify stability under concurrent requests",
            "Complete documentation audit: README, API docs, and module docstrings",
        ]),
        ("Week 11-12: v1.0 Release", [
            "Final code review sweep across all modules",
            "Tag v1.0 release with full changelog",
            "Deploy to production environment and verify all services healthy",
            "Publish project to open-source community and notify NGO partners",
        ]),
    ])

    # Code Pipeline
    pdf.section_title("Your Code Pipeline")
    pdf.body_text("As project owner your pipeline covers architecture, integration testing, PR reviews, and release management - as well as direct development when extending core systems.")

    pdf.subsection_title("Step 1: Daily Project Management")
    pdf.code_block(
        "# Check open PRs and review queue\n"
        "gh pr list --repo Climate-Vision/ClimateVision\n"
        "\n"
        "# Check CI status across all branches\n"
        "gh run list --repo Climate-Vision/ClimateVision --limit 10\n"
        "\n"
        "# View open issues\n"
        "gh issue list --repo Climate-Vision/ClimateVision --label bug"
    )

    pdf.subsection_title("Step 2: Review & Merge a Team Member's PR")
    pdf.code_block(
        "# Fetch and checkout their branch for local testing\n"
        "git fetch origin\n"
        "git checkout feature/data-sentinel2-preprocessing\n"
        "\n"
        "# Test their code runs correctly\n"
        "pip install -r requirements.txt\n"
        "python -c \"from climatevision.data.preprocessing import preprocess_tiles; print('OK')\"\n"
        "\n"
        "# Review on GitHub and approve\n"
        "gh pr review <PR_NUMBER> --approve --body \"Tested locally - preprocessing pipeline works correctly\"\n"
        "\n"
        "# Merge into develop\n"
        "gh pr merge <PR_NUMBER> --squash --delete-branch"
    )

    pdf.subsection_title("Step 3: Run End-to-End Integration Test")
    pdf.code_block(
        "# Start all services\n"
        "docker-compose up --build -d\n"
        "\n"
        "# Test the full pipeline: bbox -> prediction -> response\n"
        "curl -X POST http://localhost:8000/predict/json \\\n"
        "  -H \"Content-Type: application/json\" \\\n"
        "  -d '{\"bbox\": [-60,-15,-45,5], \"start_date\": \"2023-01-01\",\n"
        "       \"end_date\": \"2023-12-31\", \"analysis_type\": \"deforestation\"}'\n"
        "\n"
        "# Run automated integration tests\n"
        "pytest tests/integration/ -v\n"
        "\n"
        "# Verify frontend builds and loads dashboard data\n"
        "cd frontend && npm run build && npm run preview"
    )

    pdf.subsection_title("Step 4: Update System Configuration")
    pdf.code_block(
        "# Edit the master config (all analysis types, thresholds, model params)\n"
        "# File: config.yaml\n"
        "\n"
        "# Example: update deforestation alert threshold\n"
        "# deforestation:\n"
        "#   alert_threshold: 0.15  -> 0.10  (more sensitive)\n"
        "\n"
        "# Validate config loads correctly after changes\n"
        "python - <<'EOF'\n"
        "from climatevision.config import load_config\n"
        "cfg = load_config('config.yaml')\n"
        "print(f\"Analysis types: {list(cfg.keys())}\")\n"
        "EOF"
    )

    pdf.subsection_title("Step 5: Tag a Release")
    pdf.code_block(
        "# Ensure you are on the owner identity\n"
        "source team_docs/switch_user.sh gold\n"
        "\n"
        "# Merge develop into main for release\n"
        "git checkout main\n"
        "git merge develop --no-ff -m \"release: v1.0.0\"\n"
        "\n"
        "# Tag the release\n"
        "git tag -a v1.0.0 -m \"ClimateVision v1.0.0 - Deforestation, Ice Melt, Flood Detection\"\n"
        "\n"
        "# Push main and tag to GitHub\n"
        "git push origin main\n"
        "git push origin v1.0.0\n"
        "\n"
        "# Create GitHub release with changelog\n"
        "gh release create v1.0.0 \\\n"
        "  --title \"ClimateVision v1.0.0\" \\\n"
        "  --notes \"First production release. Supports deforestation, arctic ice, and flood detection.\""
    )

    pdf.subsection_title("Step 6: Direct Development (Core Systems)")
    pdf.code_block(
        "# When extending core architecture directly\n"
        "source team_docs/switch_user.sh gold\n"
        "\n"
        "git checkout develop && git pull origin develop\n"
        "git checkout -b feature/core-new-analysis-type\n"
        "\n"
        "# Make changes to core modules (analysis/, config.py, db.py, api/main.py)\n"
        "\n"
        "git add src/climatevision/analysis/\n"
        "git add config.yaml\n"
        "git commit -m \"feat(core): add drought detection analysis type to registry\"\n"
        "\n"
        "# Push as project owner\n"
        "git push origin feature/core-new-analysis-type"
    )

    pdf.section_title("Your Key Collaborators")
    pdf.bullet("Victor Mbachu (@cutewizzy11) - Co-owner for infrastructure decisions. Coordinate on Dockerfile, CI/CD pipelines, and production deployment architecture.")
    pdf.bullet("Edoh-Onuh (@edoh-Onuh) - ML Lead. Final authority on model architecture decisions sits with you, but Edoh drives the implementation. Review all model PRs carefully.")
    pdf.bullet("Olufemi Taiwo (femi23) - API Lead. You are the original author of main.py. Any structural changes to the API must go through your review.")
    pdf.bullet("Adeolu Mary Oshadare (@Oshgig) - Data Pipeline Lead. You built the GEE scripts she extends. Maintain alignment on data contracts between ingestion and training.")
    pdf.bullet("Francis Umo (@franchaise) - Analytics Lead. Carbon estimates and impact reports are the primary stakeholder-facing output. Review these deliverables closely.")
    pdf.bullet("Godswill Chukwu - ML Insights Lead. His experiment results and KPI reports inform your architectural and model selection decisions.")

    pdf.output(os.path.join(OUTPUT_DIR, "Gold_Okpa_Role.pdf"))
    print("Created: Gold_Okpa_Role.pdf")


if __name__ == "__main__":
    create_adeolu_doc()
    create_francis_doc()
    create_olufemi_doc()
    create_edoh_doc()
    create_victor_doc()
    create_godswill_doc()
    create_paul_doc()
    create_gold_doc()
    print(f"\nAll 8 role documents generated in: {OUTPUT_DIR}")
