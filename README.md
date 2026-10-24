# Municipal Operations & Alert Dispatch Platform (Backend Engine)

![FastAPI](https://img.shields.io/badge/FastAPI-v4.5.0-009688.svg) ![OpenCV](https://img.shields.io/badge/OpenCV-YuNet-5C3EE8.svg) ![Status](https://img.shields.io/badge/Deployment-Ready-success.svg)

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
* **Auto-Seed Demo Data:** Automatically pre-loaded on boot with 4 realistic civic hazard cases (Electricity, Water Clogging, Tree Debris, Pothole) with real photos, GPS, and risk scores.

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

## 3. Complete Frontend Integration API Endpoints

| Category | Method | Endpoint | Description |
| :--- | :--- | :--- | :--- |
| **Citizen Intake** | `POST` | `/api/v1/reports` | Ingests citizen report (multipart form: `text`, `hazard_type`, `lat`, `lng`, `image`) |
| **Public Feed / Map** | `GET` | `/api/v1/reports` | List of all public reports for citizen map/feed (coarse GPS, redacted photos) |
| **Public Tracking** | `GET` | `/api/v1/public/reports/{id}` | Detailed citizen tracking view for a specific ticket |
| **Public Media** | `GET` | `/api/v1/public/media/{filename}` | Serves the redacted JPG for citizen display |
| **Admin Stats** | `GET` | `/api/v1/admin/dashboard-stats` | KPI summary cards: total reports, pending alerts, dispatched, critical count |
| **Admin Inbox** | `GET` | `/api/v1/admin/unassigned-tickets` | Admin Inbox: Tickets waiting for alert dispatch with pre-filled department |
| **Departments** | `GET` | `/api/v1/admin/departments` | Returns list of statutory municipal departments with SLA policies |
| **1-Click Dispatch** | `POST` | `/api/v1/admin/send-alert` | Dispatches rich alert payload to concerned department |
| **Reassign / Anti-Loop** | `POST` | `/api/v1/admin/reassign-alert` | Reassigns ticket & intercepts Ping-Pong reassignment loops |
| **Officer Vault** | `GET` | `/api/v1/official/reports/{id}/raw-evidence` | Restricted Officer Vault view (Requires `X-Officer-Token: MUNICIPAL_CORP_OFFICER_2026`) |
| **Dev Tools** | `POST` | `/api/v1/admin/seed-demo-data` | Resets/re-seeds the 4 demo hazard tickets anytime |

---

## 4. Frontend Component (React + Tailwind CSS)

```tsx
import React, { useState, useEffect } from 'react';

export const MunicipalAlertConsole = () => {
  const [stats, setStats] = useState(null);
  const [tickets, setTickets] = useState([]);
  const [selectedDept, setSelectedDept] = useState({});

  useEffect(() => {
    // 1. Load Dashboard KPI Stats
    fetch('http://localhost:8000/api/v1/admin/dashboard-stats')
      .then(r => r.json())
      .then(setStats);

    // 2. Load Pending Complaints
    fetch('http://localhost:8000/api/v1/admin/unassigned-tickets')
      .then(r => r.json())
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
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900">Municipal Alert Dispatch Console</h1>
        {stats && (
          <div className="flex gap-3 text-sm">
            <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded font-semibold">Total: {stats.total_complaints}</span>
            <span className="bg-amber-100 text-amber-800 px-3 py-1 rounded font-semibold">Pending: {stats.pending_alerts}</span>
            <span className="bg-red-100 text-red-800 px-3 py-1 rounded font-semibold">Critical: {stats.critical_priority_hazards}</span>
          </div>
        )}
      </div>

      <div className="grid gap-4">
        {tickets.map(t => (
          <div key={t.ticket_id} className="p-4 border rounded shadow-sm bg-white flex gap-4">
            <img src={`http://localhost:8000${t.public_image_url}`} className="w-36 h-28 object-cover rounded border" />
            <div className="flex-1 space-y-2">
              <div className="flex justify-between items-start">
                <span className={`px-2 py-0.5 text-xs font-bold rounded ${
                  t.risk_score >= 90 ? 'bg-red-100 text-red-800' : 'bg-amber-100 text-amber-800'
                }`}>
                  Risk Score: {t.risk_score}/100 ({t.suggested_priority})
                </span>
                <span className="text-xs text-gray-500 font-mono">{t.ticket_id}</span>
              </div>
              <h3 className="font-bold text-gray-900">{t.detected_problem}</h3>
              <p className="text-sm text-gray-600">{t.sanitized_description}</p>
              
              <div className="pt-2 flex items-center gap-3">
                <label className="text-xs font-semibold text-gray-500">Concerned Department:</label>
                <select 
                  value={selectedDept[t.ticket_id]} 
                  onChange={e => setSelectedDept({...selectedDept, [t.ticket_id]: e.target.value})}
                  className="border rounded px-2 py-1 text-sm font-medium bg-gray-50"
                >
                  <option value="DEPT_ELECTRICITY">Electricity & Street Lighting (ELEC)</option>
                  <option value="DEPT_DRAINAGE">Stormwater Drainage (SWD)</option>
                  <option value="DEPT_ROADS">Roads & Infrastructure (ROADS)</option>
                  <option value="DEPT_HORTICULTURE">Horticulture & Tree Authority (HORT)</option>
                </select>
                <button 
                  onClick={() => sendAlert(t)} 
                  className="ml-auto bg-blue-600 hover:bg-blue-700 text-white font-semibold px-4 py-1.5 text-sm rounded shadow transition"
                >
                  Send Alert with Details
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
```
