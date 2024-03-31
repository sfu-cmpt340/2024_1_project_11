import hashlib

def compute_signal_hash(row):
  row_str = str(row.values)
  row_bytes = row_str.encode()
  hash_object = hashlib.sha256(row_bytes)
  return hash_object.hexdigest()[:9]