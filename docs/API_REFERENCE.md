# ClimateVision API Reference

This document provides a complete reference for the ClimateVision REST API.

## Base URL

```
http://localhost:8000/api
```

## Authentication

For organization-specific endpoints, use API key authentication:

```bash
curl -H "X-API-Key: your_api_key" http://localhost:8000/api/organizations/1/alerts
```

---

## Core Endpoints

### Health Check

Check API status and available analysis types.

```http
GET /api/health
```

**Response:**
```json
{
  "status": "ok",
  "version": "0.2.0",
  "analysis_types": ["deforestation", "ice_melting", "flooding"]
}
```

---

## Analysis Types

### List Analysis Types

Get all available analysis types.

```http
GET /api/analysis-types?enabled_only=true
```

**Response:**
```json
[
  {
    "name": "deforestation",
    "display_name": "Deforestation Detection",
    "description": "Monitor forest coverage and detect deforestation events",
    "enabled": true,
    "bands": ["B04", "B03", "B02", "B08"],
    "classes": ["non_forest", "forest"]
  },
  {
    "name": "ice_melting",
    "display_name": "Arctic Ice Melting",
    "description": "Monitor sea ice extent and melting patterns",
    "enabled": true,
    "bands": ["B02", "B03", "B04", "B11"],
    "classes": ["open_water", "sea_ice", "land"]
  }
]
```

### Get Analysis Type Details

```http
GET /api/analysis-types/{type_name}
```

**Example:** `GET /api/analysis-types/deforestation`

---

## Prediction Endpoints

### Run Prediction (JSON)

Run analysis using bounding box and date range.

```http
POST /api/predict
Content-Type: application/json

{
  "kind": "bbox",
  "analysis_type": "deforestation",
  "bbox": [-62.0, -3.1, -61.8, -2.9],
  "start_date": "2024-01-01",
  "end_date": "2024-12-31"
}
```

**Response:**
```json
{
  "run_id": 1,
  "result": {
    "analysis_type": "deforestation",
    "region": {
      "bbox": [-62.0, -3.1, -61.8, -2.9],
      "date_range": "2024-01-01 to 2024-12-31"
    },
    "ndvi_stats": {
      "NDVI_min": 0.123,
      "NDVI_mean": 0.567,
      "NDVI_max": 0.892
    },
    "inference": {
      "image_size": [256, 256],
      "forest_pixels": 45678,
      "non_forest_pixels": 19858,
      "forest_percentage": 69.72,
      "mean_confidence": 0.87
    }
  }
}
```

### Run Prediction (File Upload)

Upload satellite imagery for analysis.

```http
POST /api/predict/upload
Content-Type: multipart/form-data

kind=upload
analysis_type=ice_melting
bbox=[-73, 60, -12, 84]
start_date=2024-06-01
end_date=2024-08-31
file=@satellite_image.tif
```

**Response:**
```json
{
  "run_id": 2,
  "result": {
    "analysis_type": "ice_melting",
    "region": {
      "bbox": [-73, 60, -12, 84]
    },
    "inference": {
      "image_size": [512, 512],
      "ice_pixels": 150000,
      "water_pixels": 80000,
      "land_pixels": 32144,
      "ice_percentage": 65.2,
      "ice_extent_km2": 45000.5,
      "mean_confidence": 0.82
    }
  }
}
```

---

## Run History

### List Runs

Get analysis run history with optional filters.

```http
GET /api/runs?limit=50&status=completed&analysis_type=deforestation
```

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `limit` | int | Max results (default: 50, max: 200) |
| `status` | string | Filter by status: pending, running, completed, failed |
| `analysis_type` | string | Filter by analysis type |

**Response:**
```json
[
  {
    "id": 1,
    "kind": "bbox",
    "status": "completed",
    "analysis_type": "deforestation",
    "bbox": "[-62.0, -3.1, -61.8, -2.9]",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "created_at": "2024-12-15T10:30:00Z",
    "updated_at": "2024-12-15T10:30:45Z"
  }
]
```

### Get Run Details

```http
GET /api/runs/{run_id}
```

**Response:**
```json
{
  "run": {
    "id": 1,
    "kind": "bbox",
    "status": "completed",
    "analysis_type": "deforestation",
    "created_at": "2024-12-15T10:30:00Z"
  },
  "result": {
    "id": 1,
    "run_id": 1,
    "payload": { ... },
    "mask_path": null,
    "created_at": "2024-12-15T10:30:45Z"
  }
}
```

---

## Organization (NGO) Endpoints

### Create Organization

Register a new organization to receive alerts.

```http
POST /api/organizations
Content-Type: application/json

{
  "name": "Rainforest Alliance",
  "type": "ngo",
  "description": "Protecting rainforests worldwide",
  "contact_email": "alerts@rainforest.org",
  "website_url": "https://rainforest.org"
}
```

**Response:**
```json
{
  "id": 1,
  "name": "Rainforest Alliance",
  "type": "ngo",
  "api_key": "cv_abc123...",
  "active": true,
  "created_at": "2024-12-15T10:00:00Z"
}
```

> **Important:** Save the `api_key` securely. It cannot be retrieved later.

### List Organizations

```http
GET /api/organizations?type=ngo&limit=50
```

### Get Organization

```http
GET /api/organizations/{org_id}
```

---

## Subscriptions

Subscriptions allow organizations to monitor specific regions.

### Create Subscription

```http
POST /api/organizations/{org_id}/subscriptions
Content-Type: application/json

{
  "name": "Amazon Watch Zone 1",
  "bbox": [-62.0, -3.1, -61.8, -2.9],
  "analysis_types": ["deforestation", "wildfire"],
  "alert_threshold": 5.0,
  "notification_channel": "webhook",
  "webhook_url": "https://example.org/webhooks/climate"
}
```

**Response:**
```json
{
  "id": 1,
  "organization_id": 1,
  "name": "Amazon Watch Zone 1",
  "bbox": [-62.0, -3.1, -61.8, -2.9],
  "analysis_types": ["deforestation", "wildfire"],
  "alert_threshold": 5.0,
  "notification_channel": "webhook",
  "active": true,
  "created_at": "2024-12-15T11:00:00Z"
}
```

### List Subscriptions

```http
GET /api/organizations/{org_id}/subscriptions
```

---

## Alerts

### List Alerts

```http
GET /api/organizations/{org_id}/alerts?unacknowledged_only=true&limit=50
```

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `undelivered_only` | bool | Only undelivered alerts |
| `unacknowledged_only` | bool | Only unacknowledged alerts |
| `limit` | int | Max results |

**Response:**
```json
[
  {
    "id": 1,
    "organization_id": 1,
    "alert_type": "deforestation_detected",
    "severity": "high",
    "title": "Deforestation Detected",
    "message": "Forest loss detected: 7.5% reduction in coverage",
    "delivered": true,
    "acknowledged": false,
    "created_at": "2024-12-15T12:00:00Z"
  }
]
```

### Create Alert

```http
POST /api/organizations/{org_id}/alerts
Content-Type: application/json

{
  "alert_type": "deforestation_detected",
  "severity": "high",
  "title": "Deforestation Alert",
  "message": "Significant forest loss detected in monitored region",
  "subscription_id": 1,
  "run_id": 5
}
```

### Acknowledge Alert

```http
POST /api/alerts/{alert_id}/acknowledge
Content-Type: application/json

{
  "acknowledged_by": "analyst@rainforest.org"
}
```

### Mark Alert Delivered

```http
POST /api/alerts/{alert_id}/deliver
```

---

## Error Responses

All errors return a JSON response:

```json
{
  "detail": "Error message here"
}
```

**Common HTTP Status Codes:**
| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Resource doesn't exist |
| 422 | Validation Error - Invalid request body |
| 500 | Internal Server Error |

---

## Python SDK Example

```python
import requests

API_BASE = "http://localhost:8000/api"

# Run deforestation analysis
response = requests.post(
    f"{API_BASE}/predict",
    json={
        "kind": "bbox",
        "analysis_type": "deforestation",
        "bbox": [-62.0, -3.1, -61.8, -2.9],
        "start_date": "2024-01-01",
        "end_date": "2024-12-31"
    }
)
result = response.json()
print(f"Forest coverage: {result['result']['inference']['forest_percentage']}%")

# Create an organization
org_response = requests.post(
    f"{API_BASE}/organizations",
    json={
        "name": "My NGO",
        "type": "ngo",
        "contact_email": "contact@myngo.org"
    }
)
org = org_response.json()
api_key = org["api_key"]  # Save this!

# Create a subscription
sub_response = requests.post(
    f"{API_BASE}/organizations/{org['id']}/subscriptions",
    json={
        "name": "Amazon Region",
        "bbox": [-70, -10, -50, 5],
        "analysis_types": ["deforestation"],
        "alert_threshold": 5.0
    }
)
```

---

## JavaScript/TypeScript Example

```typescript
const API_BASE = 'http://localhost:8000/api';

// Run ice melting analysis
async function analyzeIce() {
  const response = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      kind: 'bbox',
      analysis_type: 'ice_melting',
      bbox: [-73, 60, -12, 84],
      start_date: '2024-06-01',
      end_date: '2024-08-31'
    })
  });
  
  const { run_id, result } = await response.json();
  console.log(`Ice extent: ${result.inference.ice_percentage}%`);
  return result;
}

// List organization alerts
async function getAlerts(orgId: number, apiKey: string) {
  const response = await fetch(
    `${API_BASE}/organizations/${orgId}/alerts?unacknowledged_only=true`,
    {
      headers: { 'X-API-Key': apiKey }
    }
  );
  return response.json();
}
```
