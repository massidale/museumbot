-- Una riga per sessione: il record intero piu' i campi per decidere gli invii.
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  write_key TEXT NOT NULL,
  device TEXT NOT NULL,
  completed INTEGER NOT NULL,
  answered INTEGER NOT NULL,
  updated_at TEXT NOT NULL,
  created_at TEXT NOT NULL,
  received_at TEXT NOT NULL,
  writes INTEGER NOT NULL,
  record TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_created_at ON sessions (created_at);
