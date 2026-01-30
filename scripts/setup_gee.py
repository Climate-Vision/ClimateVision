"""
Quick script to authenticate Google Earth Engine and test connection

Run this first time:
    python scripts/setup_gee.py
"""
import sys

print("=" * 60)
print("Google Earth Engine Authentication Setup")
print("=" * 60)
print()

# Check if earthengine-api is installed
try:
    import ee
    print("✓ earthengine-api is installed")
except ImportError:
    print("✗ earthengine-api not found")
    print("\nInstall it with:")
    print("  pip install earthengine-api")
    sys.exit(1)

print()
print("Step 1: Authentication")
print("-" * 60)
print("This will open your browser to authenticate with Google.")
print("You only need to do this ONCE per machine.")
print()

try:
    # Authenticate
    ee.Authenticate()
    print("✓ Authentication successful!")
except Exception as e:
    print(f"✗ Authentication failed: {e}")
    sys.exit(1)

print()
print("Step 2: Testing Connection")
print("-" * 60)

# Ask for project ID
project_id = input("Enter your GEE Project ID (from registration): ").strip()

if not project_id:
    print("✗ Project ID is required")
    sys.exit(1)

try:
    # Initialize
    ee.Initialize(project=project_id)
    print(f"✓ Initialized with project: {project_id}")
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    print("\nMake sure:")
    print("  1. Project ID is correct")
    print("  2. Earth Engine is enabled for your project")
    print("  3. You have access to the project")
    sys.exit(1)

print()
print("Step 3: Testing Query")
print("-" * 60)

try:
    # Test query - count Sentinel-2 images in Amazon
    point = ee.Geometry.Point([-62.0, -3.0])
    collection = (ee.ImageCollection('COPERNICUS/S2_SR')
        .filterBounds(point)
        .filterDate('2024-01-01', '2024-12-31')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
    
    count = collection.size().getInfo()
    print(f"✓ Found {count} Sentinel-2 images in Amazon region")
    print("✓ Earth Engine query successful!")
    
except Exception as e:
    print(f"✗ Query failed: {e}")
    sys.exit(1)

print()
print("=" * 60)
print("🎉 SUCCESS! Google Earth Engine is ready to use!")
print("=" * 60)
print()
print("Next steps:")
print("1. Add project ID to .env file:")
print(f"   echo 'GEE_PROJECT_ID={project_id}' >> .env")
print()
print("2. Test with ClimateVision:")
print("   python -c 'from climatevision.data.gee_loader import quick_load_example; quick_load_example()'")
print()
print("3. Start downloading data!")
print("   See: docs/GEE_SETUP.md")
print()
