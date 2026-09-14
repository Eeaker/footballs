CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS tracking;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS identity;

CREATE TABLE IF NOT EXISTS core.projects (
  id text NOT NULL,
  kind text NOT NULL,
  status text NOT NULL,
  payload jsonb NOT NULL,
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL,
  PRIMARY KEY (id)
) PARTITION BY HASH (id);

DO $$
BEGIN
  FOR i IN 0..7 LOOP
    EXECUTE format('CREATE TABLE IF NOT EXISTS core.projects_p%s PARTITION OF core.projects FOR VALUES WITH (modulus 8, remainder %s)', i, i);
  END LOOP;
END $$;

CREATE TABLE IF NOT EXISTS tracking.observations (
  project_id text NOT NULL,
  frame_index integer NOT NULL,
  global_id bigint NOT NULL,
  bbox real[] NOT NULL,
  confidence real,
  pitch_xy real[],
  PRIMARY KEY (project_id, frame_index, global_id)
) PARTITION BY HASH (project_id);

CREATE TABLE IF NOT EXISTS analytics.events (
  project_id text NOT NULL,
  event_id text NOT NULL,
  frame_index integer NOT NULL,
  event_type text NOT NULL,
  payload jsonb NOT NULL,
  PRIMARY KEY (project_id, event_id)
) PARTITION BY HASH (project_id);

CREATE TABLE IF NOT EXISTS identity.fragments (
  project_id text NOT NULL,
  source_id bigint NOT NULL,
  canonical_id bigint,
  state text NOT NULL CHECK (state IN ('confirmed_merged','isolated','quarantine')),
  evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
  PRIMARY KEY (project_id, source_id)
) PARTITION BY HASH (project_id);

CREATE TABLE IF NOT EXISTS identity.players (
  project_id text NOT NULL,
  canonical_id bigint NOT NULL,
  team_id text,
  jersey_number text,
  display_id text,
  source_ids bigint[] NOT NULL,
  avatar_object_key text,
  PRIMARY KEY (project_id, canonical_id)
) PARTITION BY HASH (project_id);

CREATE TABLE IF NOT EXISTS analytics.artifacts (
  project_id text NOT NULL,
  object_key text NOT NULL,
  media_type text,
  size_bytes bigint,
  sha256 text,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (project_id, object_key)
) PARTITION BY HASH (project_id);

DO $$
BEGIN
  FOR i IN 0..7 LOOP
    EXECUTE format('CREATE TABLE IF NOT EXISTS tracking.observations_p%s PARTITION OF tracking.observations FOR VALUES WITH (modulus 8, remainder %s)', i, i);
    EXECUTE format('CREATE TABLE IF NOT EXISTS analytics.events_p%s PARTITION OF analytics.events FOR VALUES WITH (modulus 8, remainder %s)', i, i);
    EXECUTE format('CREATE TABLE IF NOT EXISTS analytics.artifacts_p%s PARTITION OF analytics.artifacts FOR VALUES WITH (modulus 8, remainder %s)', i, i);
    EXECUTE format('CREATE TABLE IF NOT EXISTS identity.fragments_p%s PARTITION OF identity.fragments FOR VALUES WITH (modulus 8, remainder %s)', i, i);
    EXECUTE format('CREATE TABLE IF NOT EXISTS identity.players_p%s PARTITION OF identity.players FOR VALUES WITH (modulus 8, remainder %s)', i, i);
  END LOOP;
END $$;
