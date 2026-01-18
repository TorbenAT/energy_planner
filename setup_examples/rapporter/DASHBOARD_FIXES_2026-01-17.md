# Dashboard Ændringer - EV Ladetimer

**Dato**: 17. januar 2026, kl. 16:00  
**Formål**: Vise KUN de billigste ladetimer i stedet for hele vinduet

---

## 🎯 Problem Løst

**Før**: Dashboard viste "18:00-05:00" (hele EV-vinduet)  
**Efter**: Dashboard viser "03:00-05:59" (kun billigste timer)

---

## ✅ Gennemførte Ændringer

### 1. Fjernet "Opdater Plan" Knapper ❌
**Begrundelse**: Knapppen fungerede ikke og var unødvendig

**Fjernet**:
- Top-knap "🔄 Opdater Plan"
- Knap i "[9] EV & Opdatering" sektion

### 2. Ny "⚡ Billigste EV Ladetimer" Box ✅
**Placering**: Øverst på siden (tidligere "Næste handling")

**Funktionalitet**:
- Finder ALLE timer hvor planen lader EV (ev_kwh > 0.1)
- Sorterer efter pris (billigst først)
- Viser top 4 billigste timer
- Beregner tidsspan: fra første til sidste billige time

**Eksempel output**:
```
🔌 Tilslut EV mellem 03:00-05:59

Billigste timer:
- 03:00: 1.33 DKK/kWh (7.4 kWh)
- 04:00: 1.35 DKK/kWh (7.4 kWh)
- 05:00: 1.38 DKK/kWh (7.4 kWh)
- 02:00: 1.42 DKK/kWh (7.3 kWh)

Total: 29.5 kWh
```

### 3. Opdateret "[9b] EV Ladeplan Detaljer" ✅
**Før**: Viste "Næste EV vindue & Noter" med generisk info  
**Efter**: Viser konkret ladeplan sorteret efter pris

**Ny funktionalitet**:
- Status (connected/disconnected)
- Planlagt total ladning (kWh)
- Advarsel hvis ingen ladning planlagt
- Top 8 billigste timer sorteret efter pris
- Dato + tidspunkt + pris + energi

**Eksempel**:
```
Status: Disconnected
⚠️ Ingen ladning planlagt (bil disconnected)

🎯 Bedste ladetimer (sorteret efter pris):
- 01-18 03:00: 1.33 DKK/kWh → 7.4 kWh
- 01-18 04:00: 1.35 DKK/kWh → 7.4 kWh
- 01-18 05:00: 1.38 DKK/kWh → 7.4 kWh
...
```

### 4. Fjernet "[10b] Pris & arbitrage tabel" ❌
**Begrundelse**: Viste "Ingen prisdata tilgængelig" - sensor findes ikke

**Løsning**: Helt fjernet da "[10] Arbitrage & Vent" tabel dækker samme behov

---

## 🔧 Teknisk Implementation

### Billigste Timer Logik
```jinja2
{% set plan = state_attr('sensor.energy_plan', 'plan') or [] %}
{% set ev_slots = namespace(items=[]) %}

# Saml alle slots med EV-ladning
{% for slot in plan[:72] %}
  {% if slot[10] > 0.1 %}  # slot[10] = ev_charge_kwh
    {% set ev_slots.items = ev_slots.items + [
      {'time': slot[0], 'price': slot[1], 'ev_kwh': slot[10]}
    ] %}
  {% endif %}
{% endfor %}

# Sorter efter pris
{% set sorted_slots = ev_slots.items | sort(attribute='price') %}
{% set top_slots = sorted_slots[:4] %}  # Tag 4 billigste

# Beregn tidsspan
{% set first_time = top_slots[0].time.split('T')[1].split(':')[0] + ':00' %}
{% set last_time = top_slots[-1].time.split('T')[1].split(':')[0] + ':59' %}
```

### Array Indices i sensor.energy_plan.plan
```
slot[0] = timestamp
slot[1] = price_buy
slot[10] = ev_charge_kwh
```

---

## 📸 Forventet Resultat

### Øverste Box: "⚡ Billigste EV Ladetimer"
```
🔌 Tilslut EV mellem 03:00-05:59

Billigste timer:
- 03:00: 1.33 DKK/kWh (7.4 kWh)
- 04:00: 1.35 DKK/kWh (7.4 kWh)  
- 05:00: 1.38 DKK/kWh (7.4 kWh)
- 02:00: 1.42 DKK/kWh (7.3 kWh)

Total: 29.5 kWh
```

### Sektion [9b]: "EV Ladeplan Detaljer"
```
Status: Disconnected
⚠️ Ingen ladning planlagt (bil disconnected)

🎯 Bedste ladetimer (sorteret efter pris):
- 01-18 02:00: 1.42 DKK/kWh → 7.3 kWh
- 01-18 03:00: 1.33 DKK/kWh → 7.4 kWh
- 01-18 04:00: 1.35 DKK/kWh → 7.4 kWh
- 01-18 05:00: 1.38 DKK/kWh → 7.4 kWh
- 01-18 01:00: 1.50 DKK/kWh → 7.4 kWh
...
```

---

## 🔄 Sådan Aktiveres

Dashboard-ændringer træder i kraft **øjeblikkeligt** (ingen genstart nødvendig).

**Refresh browser**:
1. Åbn https://home.andsbjerg.dk/energy-planner/optimering
2. Tryk Ctrl+F5 (hard refresh)
3. Tjek at "⚡ Billigste EV Ladetimer" vises øverst

---

## 🎓 Forklaring På Logikken

### Hvorfor Vises 03:00-05:59 Nu?

**Planens data** (eksempel baseret på dit screenshot):
```
Alle EV-slots i planen:
18:00: 1.85 DKK → 0 kWh (dyrt, ingen ladning)
01:00: 1.50 DKK → 7.4 kWh
02:00: 1.42 DKK → 7.3 kWh
03:00: 1.33 DKK → 7.4 kWh ← BILLIGST
04:00: 1.35 DKK → 7.4 kWh
05:00: 1.38 DKK → 7.4 kWh
```

**Top 4 billigste**:
1. 03:00 (1.33 DKK)
2. 04:00 (1.35 DKK)
3. 05:00 (1.38 DKK)
4. 02:00 (1.42 DKK)

**Tidsspan**: Første (03:00) til sidste (05:00) → **03:00-05:59**

### Hvorfor Ikke Bare 16:00-07:00?

Planen spreder ladningen over vinduet, men koncentrerer energien i de **billigste timer**.

**Gammel visning**: "18:00-05:00" (hele vinduet)  
→ Misvisende - du skal ikke tilslutte kl. 18

**Ny visning**: "03:00-05:59" (kun billige timer)  
→ Præcis - tilslut i dette tidsrum for optimal pris

---

## ⚠️ Vigtig Note

Dashboard viser kun **planlagte** timer. Hvis bilen er **disconnected**:
- Planen laver stadig beregninger
- Men faktisk ladning = 0 kWh
- Dashboard viser advarsel: "⚠️ Ingen ladning planlagt (bil disconnected)"

**Løsning**: Tilslut bil inden kl. 03:00 i nat.

---

**Fil opdateret**: [energy_planner_dashboard.yaml](z:\dashboards\energy_planner_dashboard.yaml)  
**Test URL**: https://home.andsbjerg.dk/energy-planner/optimering  
**Refresh**: Ctrl+F5 i browser
