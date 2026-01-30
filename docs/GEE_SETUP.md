# Google Earth Engine Setup Guide

## ✅ You've Registered - Great!

Now you need to authenticate and start using it.

---

## 🔐 Authentication Methods

### Method 1: OAuth (Recommended for Development)

#### Step 1: Authenticate Once
```bash
# Install Earth Engine
pip install earthengine-api

# Run authentication (opens browser)
python -c "import ee; ee.Authenticate()"
```

This will:
1. Open your browser
2. Ask you to sign in to Google
3. Grant permissions to Earth Engine
4. Save credentials locally

**You only need to do this ONCE per machine.**

#### Step 2: Test It Works
```bash
python -c "import ee; ee.Initialize(project='YOUR_PROJECT_ID'); print('✓ Success!')"
```

Replace `YOUR_PROJECT_ID` with your actual project ID from the registration.

---

### Method 2: Service Account (For Production Later)

**Use this when deploying to servers.** Not needed now.

1. Create service account in Google Cloud Console
2. Download JSON key file
3. Store securely (never commit to git!)
4. Initialize with:
```python
import ee
credentials = ee.ServiceAccountCredentials(email, key_file)
ee.Initialize(credentials)
```

---

## 🚀 Quick Test

### Test 1: Basic Connection
```python
import ee

# Initialize (use your project ID)
ee.Initialize(project='your-project-id')

# Test query
point = ee.Geometry.Point([-62.0, -3.0])
image = ee.ImageCollection('COPERNICUS/S2_SR').filterBounds(point).first()

print("✓ Earth Engine is working!")
print(f"Image ID: {image.get('system:index').getInfo()}")
```

### Test 2: Use ClimateVision Loader
```python
from climatevision.data.gee_loader import quick_load_example

# This tests your GEE connection
quick_load_example()
```

Expected output:
```
Testing Google Earth Engine connection...
✓ Found 156 images matching criteria
✓ Google Earth Engine is working!
```

---

## 📝 Store Your Project ID

### Create `.env` file (DON'T commit this!)
```bash
# In project root
echo "GEE_PROJECT_ID=your-actual-project-id" > .env
```

### Use in Code
```python
import os
from dotenv import load_dotenv

load_dotenv()
project_id = os.getenv('GEE_PROJECT_ID')
```

---

## 🛠️ When to Use It

### NOW (Week 2-3):
**Engineer 2** will use your credentials to:
- Build the data loader (`data/gee_loader.py`) ✅ Template created
- Download sample Sentinel-2 imagery
- Create preprocessing pipeline
- Test with real forest data

### DON'T WORRY YET ABOUT:
- Production deployment
- Service accounts
- Scaling to millions of images
- Cost optimization

Just get it working locally first!

---

## 🎯 Next Steps for Engineer 2

With GEE working, Engineer 2 should:

### Week 2 Tasks:
1. **Complete `gee_loader.py`**:
   - Implement `load_sentinel2()` download
   - Add `geemap` for easy downloads
   - Test on 10 sample locations

2. **Create dataset class**:
   ```python
   # data/dataset.py
   class GEEForestDataset(Dataset):
       def __init__(self, locations, date_range):
           self.loader = GEEDataLoader()
           # Load images for each location
   ```

3. **Download sample data**:
   - 100 images from Amazon
   - 100 images from Congo Basin
   - 100 images from Southeast Asia

4. **Preprocess pipeline**:
   - Normalize bands
   - Create 256x256 tiles
   - Generate masks (if labels available)

---

## 📦 Additional Tools Needed

Install these for working with GEE:
```bash
pip install earthengine-api
pip install geemap  # Makes downloading easier
pip install folium  # For visualization
```

### Useful GEE Tools:
```python
import geemap

# Download image as numpy array
image_np = geemap.ee_to_numpy(ee_image)

# Display on interactive map
Map = geemap.Map()
Map.addLayer(ee_image, {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000})
Map.centerObject(geometry, zoom=10)
Map
```

---

## 🐛 Troubleshooting

### Error: "User credentials not found"
```bash
# Re-authenticate
python -c "import ee; ee.Authenticate()"
```

### Error: "Project not found"
```bash
# Check project ID at: https://console.cloud.google.com/
# Make sure Earth Engine is enabled for that project
```

### Error: "Quota exceeded"
- Free tier has limits
- Use `.median()` or `.mosaic()` to reduce data
- Process smaller areas first

### Rate limiting:
```python
# Add this for large jobs
import time
time.sleep(1)  # Between requests
```

---

## 📚 Learning Resources

**Official Docs:**
- https://developers.google.com/earth-engine/
- https://developers.google.com/earth-engine/tutorials/community/intro-to-python-api

**Sentinel-2 Specific:**
- https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR

**Example Notebooks:**
- https://github.com/giswqs/earthengine-py-notebooks

---

## ✅ Summary

**You need to:**
1. ✅ Run `ee.Authenticate()` once
2. ✅ Add your project ID to `.env`
3. ✅ Test with `quick_load_example()`
4. ✅ Share credentials with Engineer 2 (securely!)

**You DON'T need to:**
- ❌ Add credentials to GitHub
- ❌ Set up service accounts yet
- ❌ Worry about production deployment
- ❌ Pay anything (free tier is fine for now)

---

## 🚀 Ready to Build!

Once authenticated, Engineer 2 can start downloading real satellite data and building the data pipeline!

**Questions?** Check the GEE documentation or open a GitHub issue.
