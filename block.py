import pandas as pd
import sys
import dask.dataframe as dd
import simple_icd_10 as icd
import icd10
import simple_icd_10_cm as cm
import re

print("read csv")
df = pd.read_csv("data/cleaned_data.csv.gz")
print("finsiedh reading csv")

# Pre-compile a regex for a valid ICD-10 base (letter + 2 digits)
_VALID_BASE = re.compile(r'^[A-Z][0-9]{2}$')

def remove_aa_from_start(txt):
    if txt[0] == 'å' and txt[1] == 'å' and txt[3] == 'å':
        txt = txt[3:]
    elif txt[0] == 'å' and txt[1] == 'å':
        txt = txt[2:]
    elif txt[0] == 'å':
        txt = txt[1:]

    return txt

def is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


EN_DASH = "–"   # some sources use en dash, others hyphen
HYPHEN = "-"
_BLOCK_RANGE_RE = re.compile(r"^[A-TV-Z]\d{2}\s*[–-]\s*[A-TV-Z]\d{2}$")  # e.g., C00–C14 or C00-C14
_VALID_BASE = re.compile(r"^[A-TV-Z]\d{2}")  # same idea as yours

def _insert_dot_safely(icd, code):
    """Try icd.add_dot; if it fails, return code unchanged."""
    try:
        return icd.add_dot(code)
    except Exception:
        return code

def _looks_like_block_code(s):
    """Heuristic: block identifiers often look like 'C00–C14' or 'A00-A09'."""
    if not isinstance(s, str):
        return False
    s = s.strip().upper()
    s = s.replace(" ", "")
    return bool(_BLOCK_RANGE_RE.match(s))

def _get_node_kind(icd, node_id):
    """Best-effort probe to see if the library exposes a 'kind' or similar."""
    # Try a few possible APIs defensively.
    for getter in ("get_kind", "get_type", "kind", "type"):
        try:
            fn = getattr(icd, getter, None)
            if callable(fn):
                return fn(node_id)
        except Exception:
            pass
    # Some libs have a get_node(...) returning a dict
    try:
        node = icd.get_node(node_id)
        if isinstance(node, dict):
            for k in ("kind", "type", "node_type", "level"):
                if k in node:
                    return node[k]
    except Exception:
        pass
    return None

def _format_block_label(icd, block_id):
    """Return 'BLOCKCODE Block Title' if possible, else just description."""
    try:
        title = icd.get_description(block_id)
    except Exception:
        title = None

    # If the block_id isn't a range (some libs store a surrogate id),
    # try to retrieve a display code if available.
    display_code = None
    # Common possibility: get_code or get_label
    for getter in ("get_code", "get_label", "get_title"):
        try:
            fn = getattr(icd, getter, None)
            if callable(fn):
                display_code = fn(block_id)
                break
        except Exception:
            pass

    # Fallback: if the id itself looks like a range, we can display it
    if display_code is None and _looks_like_block_code(str(block_id)):
        display_code = str(block_id)

    if display_code and title:
        return f"{display_code} {title}"
    return title or str(block_id)

def get_icd10_block_number(icd10_code):
    """
    Map an ICD-10 code to its corresponding block (group) code.
    Returns something like "A00-A09" rather than the full description.
    """
    if pd.isna(icd10_code):
        return "Unknown"

    icd10_code = str(icd10_code).strip().upper()

    
    

    # Handle empty or invalid codes
    if not icd10_code or icd10_code == 'NAN' or icd10_code == "-" or is_int(icd10_code):
        return "Unknown"


    if pd.isna(icd10_code):
        return "Missing diagninosis"
    
    icd10_code = str(icd10_code).strip().upper()

    # Handle empty or invalid codes
    if not icd10_code or icd10_code == 'NAN' or icd10_code == "-":
        #if icd10_code != None:
        #    print(f'{icd10_code} missig')
        
        return "Misssing diagnosis"

    #if icd10_code == "V2S9" | icd10_code == "Y4N" | icd10_code == "Y1NX":
    #    return "V01–Y98"
    
    if len(icd10_code) > 4:
       icd10_code =icd10_code[:4] 

    try:
        icd10_code = icd.add_dot(icd10_code)
    except:
        pass

    
    if icd10_code and _VALID_BASE.match(icd10_code[:3]):
        try:
            code=icd.get_ancestors(icd10_code)
            code=code[-2]
            return code
        except:
            pass  
        

    try:
        if len(icd10_code) > 3: 
                tmp = icd10_code[:3]
                code = icd.get_ancestors(tmp)
                code = code[-2]
                return code
    except:
        pass

    if len(icd10_code) > 1:
        if icd10_code == "B59":
            return "A00–B99"

    if len(icd10_code) > 1:
        if icd10_code[0] == 'T':
            return "Injury, poisoning and certain other consequences of external causes"

    if len(icd10_code) > 1:
        if icd10_code[0] == 'X':
            return "Self Injury"

   
   
    if len(icd10_code) > 1:
        if icd10_code[0] == 'U':
            return "U00–U99 included Covid"

    if is_int(icd10_code):
        return "F00–F99"

    if is_float(icd10_code):
        return "F00–F99"

    if len(icd10_code) > 1:
        #print(type(icd10_code))
        #print(icd10_code[0])
        if icd10_code[0] == "V" or icd10_code[0] == "W" or icd10_code[0] == "Y":
                return "External causes of morbidity and mortality"


    if  icd10_code != None:
        return icd10_code
    else:
        return "Unknown diagnosis"



def get_icd10_block_name(icd10_code, icd):
    """
    Map an ICD-10 code to its corresponding *block* name (e.g., 'C15–C26 Malignant neoplasms of digestive organs').

    Parameters
    ----------
    icd10_code : str
        ICD-10 code (with or without dot).
    icd : object
        ICD helper with methods like add_dot, get_ancestors, get_description, etc.

    Returns
    -------
    str
        Block code + title when found, otherwise a sensible fallback.
    """
    # --- basic missing handling (kept from your version) ---
    if pd.isna(icd10_code):
        return "Error diagnosis missing from rdap"

    icd10_code = str(icd10_code).strip().upper()
    if not icd10_code or icd10_code in {"NAN", "-"}:
        return "Error diagnosis missing from rdap"

    # Trim long strings to something ICD-like (first 4 chars is reasonable)
    if len(icd10_code) > 4:
        icd10_code = icd10_code[:4]

    # Try to insert dot (e.g., C150 -> C15.0)
    icd10_code = _insert_dot_safely(icd, icd10_code)

    # Helper to try ancestors for a given code token
    def find_block_for(code_token):
        try:
            ancestors = icd.get_ancestors(code_token)
        except Exception:
            return None

        if not ancestors:
            return None

        # Search ancestors from closest to farthest for a 'block'
        # We’ll check: (1) explicit kind/type, (2) code format (range)
        for node_id in reversed(ancestors):  # chapter is usually earlier; leaf later. Reverse to start near leaf.
            kind = _get_node_kind(icd, node_id)
            if kind and str(kind).lower() == "block":
                return _format_block_label(icd, node_id)

        # If no explicit kind, try pattern-based detection
        for node_id in reversed(ancestors):
            if _looks_like_block_code(str(node_id)):
                return _format_block_label(icd, node_id)

            # Sometimes description itself contains the range; try to fetch it
            try:
                desc = icd.get_description(node_id)
                if desc and _looks_like_block_code(desc.split(" ")[0]):
                    # e.g., "C15–C26 Malignant neoplasms of digestive organs"
                    return desc
            except Exception:
                pass

        return None

    # --- Primary attempt: use the full (possibly dotted) code ---
    if icd10_code and _VALID_BASE.match(icd10_code[:3]):
        block = find_block_for(icd10_code)
        if block:
            return block

    # --- Secondary attempt: try with the 3-character category (e.g., C15) ---
    if len(icd10_code) >= 3:
        block = find_block_for(icd10_code[:3])
        if block:
            return block

    # --- Your domain-specific fallbacks (kept, slightly tidied) ---
    if icd10_code == "B59":
        return "A15–B99 Certain infectious and parasitic diseases"  # broaden to chapter if block not found

    if icd10_code.startswith("U"):
        return "U00–U99 Codes for special purposes (includes COVID-19)"

    if icd10_code.startswith("T"):
        return "S00–T98 Injury, poisoning and certain other consequences of external causes"

    if icd10_code.startswith("X"):
        return "X60–X84 Intentional self-harm (Self injury)"

    # If it’s a CSV like '295.30, ...' or a numeric DSM stub → mental/behavioural fallback
    if "," in icd10_code:
        dsm = icd10_code.split(",")[0].strip()
        try:
            float(dsm)
            return "F00–F99 Mental and behavioural disorders"
        except Exception:
            pass
    # Purely numeric?
    try:
        float(icd10_code)
        return "F00–F99 Mental and behavioural disorders"
    except Exception:
        pass

    if icd10_code[0] in {"V", "W", "Y"}:
        return "V01–Y98 External causes of morbidity and mortality"

    # --- Last resort: return the cleaned code so the caller can inspect/log it ---
    return icd10_code


df['ICD10_block_diagnosis']=df['HovedDiagnosekode'].apply(lambda c: get_icd10_block_name(c, icd))
df['ICD10_block_diagnosis_number']=df['HovedDiagnosekode'].apply(get_icd10_block_number)
print(df['ICD10_block_diagnosis'])
df.to_csv("data/test_data.csv.gz",index=False)
