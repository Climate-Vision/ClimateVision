# Google Earth Engine Authentication Guide

**Official Documentation**: https://developers.google.com/earth-engine/guides/auth

**Last Updated**: January 2026

---

## 📚 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Authentication Methods](#authentication-methods)
3. [Quick Start (Recommended)](#quick-start)
4. [Service Accounts (Production)](#service-accounts)
5. [Cloud Project Setup](#cloud-project-setup)
6. [Troubleshooting](#troubleshooting)
7. [Security Best Practices](#security-best-practices)

---

## Prerequisites

### ✅ What You Need

1. **Google Account** with Earth Engine access
   - Register at: https://earthengine.google.com/signup/
   - Wait for approval (usually 1-2 days for research/non-profit)

2. **Cloud Project** (Required since May 2024)
   - Create at: https://console.cloud.google.com/
   - Enable Earth Engine API
   - Note your Project ID

3. **Python Environment**
   ```bash
   pip install earthengine-api
   ```

---

## Authentication Methods

Google Earth Engine supports multiple authentication methods:

### Comparison Table

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| **Standard OAuth** | Local development | Easy, one-time setup | Requires browser |
| **Service Account** | Production servers | Automated, secure | More setup |
| **Notebook Authenticator** | Colab/Jupyter | Built-in support | Limited to notebooks |
| **High-Volume Endpoint** | Heavy processing | Higher limits | Requires approval |

---

## Quick Start (Recommended)

### 🚀 Method 1: Standard OAuth Authentication

**Best for**: Local development, testing, team work

#### Step 1: Run Setup Script

```bash
python scripts/setup_gee.py
```

This automated script will:
1. ✓ Check dependencies
2. ✓ Run authentication (opens browser)
3. ✓ Test connection
4. ✓ Save Project ID to `.env`

#### Step 2: What Happens

**Authentication Flow**:
```python
import ee

# Opens browser, asks for Google sign-in
ee.Authenticate()

# Saves credentials to: ~/.config/earthengine/credentials
# You only do this ONCE per machine
```

**Using After Authentication**:
```python
import ee

# Just initialize with project ID
ee.Initialize(project='your-project-id')

# Now you can use Earth Engine!
point = ee.Geometry.Point([-62.0, -3.0])
collection = ee.ImageCollection('COPERNICUS/S2_SR').filterBounds(point)
print(f"Found {collection.size().getInfo()} images")
```

#### Step 3: Store Project ID

Create `.env` file (already in `.gitignore`):
```bash
echo "GEE_PROJECT_ID=your-project-id" > .env
```

Load in code:
```python
import os
from dotenv import load_dotenv

load_dotenv()
project_id = os.getenv('GEE_PROJECT_ID')

import ee
ee.Initialize(project=project_id)
```

#### Credentials Storage

After `ee.Authenticate()`, credentials are saved at:
- **Linux/Mac**: `~/.config/earthengine/credentials`
- **Windows**: `C:\Users\USERNAME\.config\earthengine\credentials`

These credentials:
- ✓ Are encrypted
- ✓ Auto-refresh when expired
- ✓ Work across all Python sessions
- ✓ Persist until you delete them

---

## Service Accounts (Production)

### 🔐 Method 2: Service Account Authentication

**Best for**: Production servers, CI/CD, Docker, automated systems

### Why Service Accounts?

✅ **No Browser Required** - Perfect for servers  
✅ **Automated** - Scripts run without human interaction  
✅ **Secure** - Separate credentials per application  
✅ **Auditable** - Track which service accessed what  

### Setup Process

#### Step 1: Create Service Account

1. Go to https://console.cloud.google.com/iam-admin/serviceaccounts
2. Select your project
3. Click **"Create Service Account"**
4. Fill in:
   - **Name**: `climatevision-gee-service`
   - **ID**: `climatevision-gee-service` (auto-generated)
   - **Description**: `Earth Engine access for ClimateVision`
5. Click **"Create and Continue"**

#### Step 2: Grant Permissions

Select role: **Earth Engine Resource Writer**

This gives permission to:
- Read Earth Engine datasets
- Export data
- Run computations

Click **"Continue"** → **"Done"**

#### Step 3: Create Private Key

1. Click on your new service account
2. Go to **"Keys"** tab
3. Click **"Add Key"** → **"Create new key"**
4. Choose **JSON** format
5. Click **"Create"**
6. Key file downloads automatically

**⚠️ CRITICAL**: This key file gives full access. Store it securely!

#### Step 4: Store Key Securely

```bash
# Create secure directory
mkdir -p ~/.gee_keys
chmod 700 ~/.gee_keys

# Move key file
mv ~/Downloads/climatevision-*.json ~/.gee_keys/service-account.json
chmod 600 ~/.gee_keys/service-account.json
```

Add to `.env`:
```bash
GEE_SERVICE_ACCOUNT_KEY=/home/user/.gee_keys/service-account.json
GEE_PROJECT_ID=your-project-id
```

#### Step 5: Use in Code

```python
import ee
import json
import os

# Load key file path from environment
key_file = os.getenv('GEE_SERVICE_ACCOUNT_KEY')
project_id = os.getenv('GEE_PROJECT_ID')

# Initialize with service account
credentials = ee.ServiceAccountCredentials(
    email=None,  # Auto-extracted from key file
    key_file=key_file
)

ee.Initialize(credentials=credentials, project=project_id)
```

**For ClimateVision production deployment**:

```python
# src/climatevision/data/gee_loader.py (production mode)
class GEEDataLoader:
    def __init__(self, use_service_account=False):
        if use_service_account:
            key_file = os.getenv('GEE_SERVICE_ACCOUNT_KEY')
            if not key_file:
                raise ValueError("GEE_SERVICE_ACCOUNT_KEY not set")
            
            credentials = ee.ServiceAccountCredentials(key_file=key_file)
            ee.Initialize(credentials=credentials, project=os.getenv('GEE_PROJECT_ID'))
        else:
            # Standard user authentication
            ee.Initialize(project=os.getenv('GEE_PROJECT_ID'))
```

---

## Cloud Project Setup

### 📦 Creating a Cloud Project

#### Step 1: Create Project

1. Go to https://console.cloud.google.com/
2. Click project dropdown → **"New Project"**
3. Enter details:
   - **Project name**: `ClimateVision Earth Engine`
   - **Project ID**: `climatevision-ee-2025` (must be unique globally)
   - **Location**: Select organization (if applicable)
4. Click **"Create"**

#### Step 2: Enable Earth Engine API

1. Go to: https://console.cloud.google.com/apis/library/earthengine.googleapis.com
2. Select your project from dropdown
3. Click **"Enable"**
4. Wait 1-2 minutes for API activation

#### Step 3: Verify Setup

```python
import ee

# This should work without errors
ee.Initialize(project='climatevision-ee-2025')

print("✓ Earth Engine initialized successfully")
```

### Project ID Format

Valid Project IDs:
- ✓ `my-project-123`
- ✓ `climatevision2025`
- ✓ `ee-username`

Invalid Project IDs:
- ✗ `My_Project` (uppercase)
- ✗ `project@name` (special chars)
- ✗ `-project` (starts with hyphen)

---

## High-Volume Endpoint

### ⚡ Method 3: High-Volume Processing

**Best for**: Processing millions of images, production workloads with high throughput

### When to Use

Use high-volume endpoint if you're:
- Processing 100,000+ images per day
- Running continuous monitoring systems
- Hitting rate limits on standard endpoint
- Building commercial applications

### Setup

```python
import ee

# Initialize with high-volume endpoint
ee.Initialize(
    project='your-project-id',
    opt_url='https://earthengine-highvolume.googleapis.com'
)
```

### Requirements

- Earth Engine Commercial license, OR
- Approved high-volume access request

Contact: earthengine-commercial@google.com

---

## Notebook Authentication

### 📓 Method 4: Google Colab / Jupyter

**Best for**: Interactive notebooks, demonstrations, tutorials

### Google Colab

```python
# Option 1: Simple authentication
import ee
ee.Authenticate()
ee.Initialize(project='your-project-id')

# Option 2: Using Colab auth
from google.colab import auth
auth.authenticate_user()
import ee
ee.Initialize(project='your-project-id')
```

### Local Jupyter

```python
import ee

# Same as standard authentication
ee.Authenticate()  # Opens browser first time
ee.Initialize(project='your-project-id')
```

---

## Troubleshooting

### Common Issues

#### 1. "Please authorize access to your Earth Engine account"

**Solution**: Run authentication
```python
import ee
ee.Authenticate()
```

#### 2. "Project not found" or "Invalid project"

**Solutions**:
- Check project ID is correct: https://console.cloud.google.com/
- Verify Earth Engine API is enabled for that project
- Ensure you have access to the project (check IAM permissions)

#### 3. "Quota exceeded"

**Solutions**:
- Wait and retry (quotas reset)
- Use `.median()` or `.mosaic()` to reduce computation
- Process smaller regions
- Consider high-volume endpoint

#### 4. "Credentials not found"

**Solutions**:
```bash
# Re-authenticate
python -c "import ee; ee.Authenticate()"

# Check credentials file exists
ls ~/.config/earthengine/credentials
```

#### 5. Service account "Permission denied"

**Solutions**:
- Verify service account has "Earth Engine Resource Writer" role
- Check key file path is correct
- Ensure key file has proper permissions (chmod 600)
- Confirm project ID matches

### Debugging Tips

```python
import ee

# Check if initialized
try:
    ee.Initialize(project='your-project-id')
    print("✓ Initialized")
except Exception as e:
    print(f"✗ Error: {e}")

# Test basic query
try:
    image_count = ee.ImageCollection('COPERNICUS/S2_SR').size().getInfo()
    print(f"✓ Query works. Image count: {image_count}")
except Exception as e:
    print(f"✗ Query failed: {e}")
```

---

## Security Best Practices

### 🔒 DO:

✅ **Use `.env` files for credentials**
```bash
# .env (in .gitignore)
GEE_PROJECT_ID=your-project-id
GEE_SERVICE_ACCOUNT_KEY=/secure/path/key.json
```

✅ **Set proper file permissions**
```bash
chmod 600 ~/.config/earthengine/credentials
chmod 600 /path/to/service-account-key.json
```

✅ **Use different service accounts per environment**
- `climatevision-dev-sa` for development
- `climatevision-prod-sa` for production

✅ **Rotate service account keys regularly**
- Create new key
- Update deployments
- Delete old key

✅ **Use secret management in production**
- AWS Secrets Manager
- Google Cloud Secret Manager
- HashiCorp Vault

### ❌ DON'T:

❌ **Never commit credentials to git**
```bash
# BAD - DON'T DO THIS
git add .env
git add service-account-key.json
```

❌ **Never hardcode credentials**
```python
# BAD - DON'T DO THIS
project_id = "my-secret-project"  # OK, project IDs aren't secret
key_path = "/path/to/key.json"    # OK if path is secure
```

❌ **Never share service account keys**
- Each team member gets their own key
- Services get separate service accounts

❌ **Never use production keys in development**
- Use separate service accounts
- Limit development key permissions

---

## For ClimateVision Team

### Current Setup (Development)

**Method**: Standard OAuth Authentication

**What each person does**:
1. Clone repository
2. Run `python scripts/setup_gee.py`
3. Enter their own Project ID
4. Work with their own credentials

**Team member credentials are**:
- ✓ Stored locally on their machine
- ✓ Not shared or committed
- ✓ Independent of each other

### Future Setup (Production)

**Method**: Service Account

**When deploying API/Dashboard**:
1. Create production service account
2. Store key in secure secret manager
3. Configure production environment
4. Monitor usage and quotas

---

## Quick Reference

### Local Development
```bash
# One-time setup
python scripts/setup_gee.py

# In every session
python -c "import ee; ee.Initialize(project='PROJECT_ID')"
```

### Production Deployment
```bash
# Set environment variables
export GEE_SERVICE_ACCOUNT_KEY=/path/to/key.json
export GEE_PROJECT_ID=your-project-id

# In code
import ee
credentials = ee.ServiceAccountCredentials(key_file=key_path)
ee.Initialize(credentials=credentials, project=project_id)
```

### Testing
```bash
# Test authentication
python -c "import ee; ee.Initialize(project='PROJECT_ID'); print('✓ Works')"

# Test query
python -c "
import ee
ee.Initialize(project='PROJECT_ID')
count = ee.ImageCollection('COPERNICUS/S2_SR').size().getInfo()
print(f'✓ Found {count} images')
"
```

---

## Additional Resources

**Official Documentation**:
- Authentication Guide: https://developers.google.com/earth-engine/guides/auth
- Python API: https://developers.google.com/earth-engine/guides/python_install
- Service Accounts: https://developers.google.com/earth-engine/guides/service_account

**ClimateVision Docs**:
- Data Loader: `src/climatevision/data/gee_loader.py`
- Setup Script: `scripts/setup_gee.py`
- Project Structure: `PROJECT_STRUCTURE.md`

**Support**:
- Earth Engine Forum: https://groups.google.com/g/google-earth-engine-developers
- Stack Overflow: Tag `google-earth-engine`
- GitHub Issues: For ClimateVision-specific problems

---

## Summary

### ✅ What You Should Do NOW

1. **Run the setup script**:
   ```bash
   python scripts/setup_gee.py
   ```

2. **Store your Project ID**:
   ```bash
   echo "GEE_PROJECT_ID=your-project-id" > .env
   ```

3. **Test it works**:
   ```python
   from climatevision.data.gee_loader import quick_load_example
   quick_load_example()
   ```

4. **Start building**:
   - Engineer 2: Complete `gee_loader.py`
   - Engineer 2: Download sample data
   - Team: Start training models!

### 🔮 What You'll Do LATER

- Set up service account for production
- Configure high-volume endpoint if needed
- Implement proper secret management
- Monitor quotas and usage

---

**Questions?** Open a GitHub issue or check the official docs!

**Ready to download satellite data!** 🛰️
