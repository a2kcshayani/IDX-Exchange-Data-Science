# Week 1 — Key Column Notes

Source: CRMLS (California Regional MLS) sold-listing extracts, `CRMLSSold*.csv`, one file per month,
delivered through the Trestle (RESO Data Dictionary) feed. Definitions below follow the RESO standard;
check each against `resources/Trestle Property MetaData.pdf` on the FTP server and fix anything that differs.

**Role** column:
- **target** = what we predict
- **feature** = allowed model input
- **filter / id / QA** = used for scoping, de-duplication, or data-quality checks only
- **excluded** = leaks the answer or doesn't exist for off-market homes

## Target and scope

| Column | Meaning | Role |
|---|---|---|
| `ClosePrice` | Final price the property sold for at closing. | **target** |
| `CloseDate` | Date the sale closed (recorded). Defines the month used for the chronological split and the valuation date for age/seasonality. | split key |
| `PropertyType` | Top-level class (Residential, Residential Income, Land, Commercial…). Keep `Residential`. | filter |
| `PropertySubType` | Finer class (SingleFamilyResidence, Condominium, Townhouse…). Keep `SingleFamilyResidence`. | filter |
| `ListingKey` / `ListingId` | System key / MLS number for the listing. Used to drop the same sale appearing in more than one file. | id |
| `UnparsedAddress` | Full street address as one string. Used with CloseDate for de-duplication. | id |

## Property characteristics (known for any address)

| Column | Meaning | Role |
|---|---|---|
| `LivingArea` | Finished living space, in square feet. Usually the strongest single predictor. | feature |
| `BedroomsTotal` | Total number of bedrooms. | feature |
| `BathroomsTotalInteger` | Total bathrooms as an integer (full + partial counted as whole). | feature |
| `LotSizeSquareFeet` | Lot area in square feet; `LotSizeAcres` is the same fact in acres (used to backfill). | feature |
| `YearBuilt` | Year construction finished. Turned into `PropertyAge` = sale year − YearBuilt. | feature (via age) |
| `GarageSpaces` | Number of cars the garage holds. | feature |
| `ParkingTotal` | Total parking spaces (garage + carport + other). | feature |
| `Stories` | Number of floors. | feature |
| `FireplacesTotal` / `FireplaceYN` | Fireplace count / presence. | feature |
| `PoolPrivateYN` | Has a private pool. | feature |
| `ViewYN` | Has a notable view. | feature |
| `AttachedGarageYN` | Garage is attached to the house. | feature |
| `NewConstructionYN` | Newly built. | feature |

## Location (fixed for any address)

| Column | Meaning | Role |
|---|---|---|
| `Latitude`, `Longitude` | Geocoded coordinates. Sanity-checked against a California bounding box. | feature |
| `CountyOrParish` | County. Low cardinality → one-hot encoded. | feature |
| `City` | City/municipality name. High cardinality → target encoded (cross-fitted). | feature |
| `PostalCode` | ZIP code, normalized to 5 digits. High cardinality → target encoded (cross-fitted). | feature |
| School district | Week 6: spatial join of coordinates to CA School District Areas 2024-25. | future feature |

## Excluded: listing/sale-process fields (leakage)

These exist only because a home was listed or sold, so the model can't use them to value an off-market
home. List prices also sit so close to ClosePrice that they inflate accuracy.

| Column | Meaning | Why excluded |
|---|---|---|
| `ListPrice` | Current asking price. | Not available off-market; near-perfect proxy for the target. |
| `OriginalListPrice` | First asking price. | Same as above. |
| `DaysOnMarket`, `CumulativeDaysOnMarket` | Days between listing and contract. | Only exists after a sale process. |
| `ListingContractDate` | Date the listing agreement was signed. | With CloseDate it rebuilds DOM. **Used only for the QA check "CloseDate before listing date".** |
| `PurchaseContractDate` | Date the buyer went into contract. | Sale-process timing. |
| List/buyer agent and office fields | Who handled the sale. | Proxy for pricing strategy, not intrinsic value. |
| Anything derived from the above | e.g. ClosePrice/ListPrice ratio, list price per sq ft, PPSF computed from ClosePrice. | Leaks the target. Use comp-based neighborhood PPSF from *prior* sales instead (Week 6). |

The structural guard for this is `idx_avm.features.audit_features`, which raises if any of these reach the model.
