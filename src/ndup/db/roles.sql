-- Least-privilege roles. ndup_worker owns the pipeline. ndup_web is read-mostly.
-- Passwords are development defaults; production setup replaces them.

DO $$
BEGIN
  CREATE ROLE ndup_worker LOGIN PASSWORD 'ndup-worker';
EXCEPTION
  WHEN duplicate_object THEN NULL;
END
$$;

DO $$
BEGIN
  CREATE ROLE ndup_web LOGIN PASSWORD 'ndup-web';
EXCEPTION
  WHEN duplicate_object THEN NULL;
END
$$;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ndup_worker;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ndup_worker;

REVOKE ALL ON ALL TABLES IN SCHEMA public FROM ndup_web;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ndup_web;
GRANT INSERT, UPDATE ON cluster_review, ignore_rule, audit_log, calibration_label TO ndup_web;
GRANT INSERT ON webhook_event, sync_task TO ndup_web;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO ndup_web;
