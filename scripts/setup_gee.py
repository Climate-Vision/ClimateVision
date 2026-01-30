"""
Google Earth Engine Authentication Setup

Official guide: https://developers.google.com/earth-engine/guides/auth

This script follows the recommended authentication flow:
1. Run ee.Authenticate() - stores credentials in ~/.config/earthengine/
2. Run ee.Initialize(project='PROJECT_ID') - initializes with your Cloud Project

You only need to authenticate ONCE per machine.
"""
import sys
import os

print("=" * 70)
print("Google Earth Engine Authentication Setup")
print("=" * 70)
print()
print("Official docs: https://developers.google.com/earth-engine/guides/auth")
print()

# Step 0: Check installation
try:
    import ee
    print("✓ earthengine-api is installed")
except ImportError:
    print("✗ earthengine-api not found")
    print("\nInstall it with:")
    print("  pip install earthengine-api")
    sys.exit(1)

print()

# Step 1: Check if already authenticated
print("Step 1: Checking existing credentials")
print("-" * 70)

credentials_exist = False
credentials_path = os.path.expanduser("~/.config/earthengine/credentials")
if os.path.exists(credentials_path):
    print(f"✓ Found existing credentials at: {credentials_path}")
    credentials_exist = True
else:
    print("○ No existing credentials found")

print()

# Step 2: Authenticate if needed
if not credentials_exist:
    print("Step 2: Authentication Required")
    print("-" * 70)
    print("This will open your browser to sign in with your Google account.")
    print("Grant Earth Engine permission to access your data.")
    print()
    input("Press Enter to continue...")
    print()
    
    try:
        # Authenticate - stores credentials locally
        ee.Authenticate()
        print("✓ Authentication successful!")
        print(f"✓ Credentials saved to: {credentials_path}")
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure you have a Google account")
        print("  2. Register for Earth Engine: https://earthengine.google.com/signup/")
        print("  3. Wait for approval email (usually 1-2 days)")
        sys.exit(1)
else:
    print("Step 2: Authentication")
    print("-" * 70)
    print("✓ Already authenticated (skipping)")
    print()

print()

# Step 3: Get Project ID
print("Step 3: Cloud Project Setup")
print("-" * 70)
print()
print("Since May 2024, a Cloud Project is REQUIRED for Earth Engine.")
print()
print("Get your Project ID from:")
print("  https://console.cloud.google.com/")
print()
print("Examples:")
print("  - ee-myusername")
print("  - my-earth-engine-project") 
print("  - climatevision-gee")
print()

# Check if already in environment
existing_project = os.getenv('GEE_PROJECT_ID')
if existing_project:
    print(f"Found existing project ID: {existing_project}")
    use_existing = input("Use this project ID? (y/n): ").strip().lower()
    if use_existing == 'y':
        project_id = existing_project
    else:
        project_id = input("Enter your Project ID: ").strip()
else:
    project_id = input("Enter your Project ID: ").strip()

if not project_id:
    print("✗ Project ID is required")
    sys.exit(1)

print()

# Step 4: Test initialization
print("Step 4: Testing Connection")
print("-" * 70)

try:
    # Initialize with project
    ee.Initialize(project=project_id)
    print(f"✓ Successfully initialized with project: {project_id}")
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    print("\nPossible issues:")
    print("  1. Project ID is incorrect")
    print("  2. Earth Engine API not enabled for this project")
    print("     Enable it at: https://console.cloud.google.com/apis/library/earthengine.googleapis.com")
    print("  3. You don't have access to this project")
    sys.exit(1)

print()

# Step 5: Test query
print("Step 5: Testing Query")
print("-" * 70)

try:
    # Test query - count Sentinel-2 images in Amazon
    print("Querying Sentinel-2 images for Amazon region...")
    point = ee.Geometry.Point([-62.0, -3.0])
    collection = (ee.ImageCollection('COPERNICUS/S2_SR')
        .filterBounds(point)
        .filterDate('2024-01-01', '2024-12-31')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
    
    count = collection.size().getInfo()
    print(f"✓ Found {count} Sentinel-2 images")
    print("✓ Earth Engine query successful!")
    
    # Get one image to verify
    if count > 0:
        first_image = collection.first()
        image_id = first_image.get('system:index').getInfo()
        print(f"✓ Sample image ID: {image_id}")
    
except Exception as e:
    print(f"✗ Query failed: {e}")
    sys.exit(1)

print()

# Step 6: Save project ID
print("Step 6: Saving Configuration")
print("-" * 70)

env_file = ".env"
if os.path.exists(env_file):
    print(f"✓ Found existing {env_file}")
    with open(env_file, 'r') as f:
        content = f.read()
    if 'GEE_PROJECT_ID' in content:
        print("✓ GEE_PROJECT_ID already in .env")
    else:
        with open(env_file, 'a') as f:
            f.write(f"\nGEE_PROJECT_ID={project_id}\n")
        print(f"✓ Added GEE_PROJECT_ID to {env_file}")
else:
    with open(env_file, 'w') as f:
        f.write(f"# ClimateVision Environment Variables\n")
        f.write(f"GEE_PROJECT_ID={project_id}\n")
    print(f"✓ Created {env_file} with GEE_PROJECT_ID")

print()
print("=" * 70)
print("🎉 SUCCESS! Google Earth Engine is fully configured!")
print("=" * 70)
print()
print("📝 What was done:")
print(f"  ✓ Authenticated (credentials in {credentials_path})")
print(f"  ✓ Verified project: {project_id}")
print(f"  ✓ Tested query successfully")
print(f"  ✓ Saved to {env_file}")
print()
print("🚀 Next steps:")
print()
print("1. Test with ClimateVision:")
print("   python -c 'from climatevision.data.gee_loader import quick_load_example; quick_load_example()'")
print()
print("2. Start downloading data:")
print("   python -c '")
print("   from climatevision.data.gee_loader import GEEDataLoader")
print("   loader = GEEDataLoader()")
print("   # Now use loader.load_sentinel2(...)")
print("   '")
print()
print("3. Read the docs:")
print("   docs/GEE_SETUP.md")
print()
print("💡 Remember:")
print("  - You only need to authenticate ONCE per machine")
print("  - Your project ID is stored in .env (already in .gitignore)")
print("  - Credentials are stored securely in ~/.config/earthengine/")
print()
