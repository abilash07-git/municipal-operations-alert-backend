# Municipal Operations & Alert Dispatch Platform (Backend Engine)`n`n![FastAPI](https://img.shields.io/badge/FastAPI-v4.5.0-009688.svg) ![OpenCV](https://img.shields.io/badge/OpenCV-YuNet-5C3EE8.svg) ![Status](https://img.shields.io/badge/Deployment-Ready-success.svg)

> **Challenge 10 (Your City, Your Services) — IMPACTX 2026**  
> Semi-Automated Civic Hazard Classification, Department Pre-Routing, Boundary Anti-Loop Deadlock Breaker & Privacy Preservation

---

## 1. Quick Start Guide

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Backend Server
```bash
python app.py
```
* **Server runs at:** `http://localhost:8000`  
* **Interactive OpenAPI Swagger Docs:** `http://localhost:8000/docs`

### Step 3: Run Integration Test
```bash
python test_client.py
```

---

## 2. Statutory Municipal Departments Configured

The platform strictly routes civic issues to authorized statutory municipal departments in India:

| # | Code | Department Name | In-Charge Officer | Official Alert Dispatch Channel | SLA Clock |
|---|---|---|---|---|---|
| 1 | **`ELEC`** | **Electricity & Street Lighting Department** | Superintending Engineer (Power & Lighting) | `electricity-emergency@municipal.gov.in` \| VHF Ch-3 \| Hotline **1912** | **1 hr** (Crit) / **3 hrs** (High) |
| 2 | **`SWD`** | **Stormwater Drainage & Flood Mitigation** | Executive Engineer (SWD & Flood Control) | `drainage-emergency@municipal.gov.in` \| VHF Ch-4 \| Flood Cell Line **1077** | **2 hrs** (Crit) / **4 hrs** (High) |
| 3 | **`ROADS`** | **Roads & Infrastructure Department** | Executive Engineer (Road Maintenance) | `roads-repair@municipal.gov.in` \| VHF Ch-2 \| Helpline **1916** | **3 hrs** (Crit) / **6 hrs** (High) |
| 4 | **`HORT`** | **Horticulture & Tree Authority** | Superintendent of Gardens & Tree Officer | `trees-emergency@municipal.gov.in` \| VHF Ch-5 \| Hotline **1099** | **2 hrs** (Crit) / **5 hrs** (High) |
| 5 | **`COMM-CTRL`**| **Municipal Commissioner's Emergency Cell** | Additional Municipal Commissioner | `commissioner-control@municipal.gov.in` \| Emergency Hotline **101** | **1 hr** (Emergency Escalation) |

---

## 3. Core Capabilities

1. **Automated Computer Vision Privacy Pipeline**:
   - Neural WIDER-FACE YuNet face detection & tight blackbox redaction.
   - Dual-palette vehicle license plate masking (white private + yellow commercial transport).
   - Strict EXIF metadata removal (prevents hardware serial or GPS leaks to public).
   - Regex masking of citizen phone numbers in description fields.

2. **Automated Hazard Classification & Pre-Routing**:
   - Classifies complaints into **Electricity/Live Wire**, **Water Clogging/Drainage**, **Road Potholes**, and **Tree Debris**.
   - Pre-selects the appropriate department in the Administrator Inbox.
   - Computes dynamic urgency risk scores (e.g., Live snapped wires score **98/100, CRITICAL**).

3. **Semi-Automated Rich Alert Dispatch**:
   - Administrator reviews pre-selected complaint with a single click.
   - Generates complete operational dispatch payload delivered via emergency channels:
     * Exact GPS coordinates for field crews + 100m coarse coordinates for public tracking.
     * Public redacted media URL vs. Official restricted evidence vault URL.
     * Dynamic SLA clock with probabilistic weather-adjusted resolution window.

4. **Anti-Ping-Pong Boundary Deadlock Breaker**:
   - If two departments bounce a complaint back and forth across a ward boundary road, the system intercepts the loop.
   - Automatically escalates the ticket directly to the Municipal Commissioner's Emergency Cell (`COMM-CTRL`) with a mandatory 1-hour SLA.

---

## 4. Frontend Integration API Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/reports` | Ingests citizen report (multipart form: `text`, `hazard_type`, `lat`, `lng`, `image`) |
| `GET` | `/api/v1/admin/unassigned-tickets` | Admin Inbox: Tickets waiting for alert dispatch with pre-filled department |
| `GET` | `/api/v1/admin/departments` | Returns list of statutory municipal departments with SLA policies |
| `POST` | `/api/v1/admin/send-alert` | 1-Click Action: Dispatches rich alert payload to concerned department |
| `POST` | `/api/v1/admin/reassign-alert` | Reassigns ticket & intercepts Ping-Pong reassignment loops |
| `GET` | `/api/v1/public/reports/{id}` | Public Citizen Tracking view (Redacted image, 100m coarse GPS) |
| `GET` | `/api/v1/official/reports/{id}/raw-evidence` | Restricted Officer Vault view (Requires `X-Officer-Token: MUNICIPAL_CORP_OFFICER_2026`) |

---

## 5. Ready-to-Use Frontend Integration Code (React)

```tsx
import React, { useState, useEffect } from 'react';

export const MunicipalAlertConsole = () => {
  const [tickets, setTickets] = useState([]);
  const [selectedDept, setSelectedDept] = useState({});

  useEffect(() => {
    fetch('http://localhost:8000/api/v1/admin/unassigned-tickets')
      .then(res => res.json())
      .then(data => {
        setTickets(data.tickets || []);
        const map = {};
        data.tickets.forEach(t => { map[t.ticket_id] = t.pre_selected_department_id; });
        setSelectedDept(map);
      });
  }, []);

  const sendAlert = async (ticket) => {
    const res = await fetch('http://localhost:8000/api/v1/admin/send-alert', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ticket_id: ticket.ticket_id,
        department_id: selectedDept[ticket.ticket_id],
        admin_id: "CHIEF_DISPATCHER",
        priority: ticket.suggested_priority,
        admin_notes: "Emergency field response order."
      })
    });
    const data = await res.json();
    alert(`Alert Dispatched to ${data.alert.concerned_department.department_name}`);
    setTickets(tickets.filter(t => t.ticket_id !== ticket.ticket_id));
  };

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-4">
      <h1 className="text-2xl font-bold">Municipal Alert Dispatch Console</h1>
      {tickets.map(t => (
        <div key={t.ticket_id} className="p-4 border rounded shadow bg-white flex gap-4">
          <img src={`http://localhost:8000${t.public_image_url}`} className="w-32 h-24 object-cover rounded" />
          <div className="flex-1">
            <h3 className="font-bold">{t.detected_problem} ({t.suggested_priority} - {t.risk_score}/100)</h3>
            <p className="text-sm text-gray-600">{t.sanitized_description}</p>
            <div className="mt-2 flex gap-2">
              <select 
                value={selectedDept[t.ticket_id]} 
                onChange={e => setSelectedDept({...selectedDept, [t.ticket_id]: e.target.value})}
                className="border rounded p-1 text-sm font-medium"
              >
                <option value="DEPT_ELECTRICITY">Electricity & Street Lighting (ELEC)</option>
                <option value="DEPT_DRAINAGE">Stormwater Drainage (SWD)</option>
                <option value="DEPT_ROADS">Roads & Infrastructure (ROADS)</option>
                <option value="DEPT_HORTICULTURE">Horticulture & Tree Authority (HORT)</option>
              </select>
              <button onClick={() => sendAlert(t)} className="bg-blue-600 text-white px-3 py-1 text-sm rounded">
                Send Alert with Details
              </button>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};
```
