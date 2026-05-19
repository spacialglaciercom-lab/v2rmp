🔒 Fix SQL injection vulnerability in query_supabase MCP tool

🎯 **What:** The `query_supabase` MCP tool allowed execution of arbitrary SQL queries from untrusted user input via `sqlx::query()`.
⚠️ **Risk:** An attacker or misbehaving AI could execute destructive queries like `DROP TABLE`, `DELETE FROM`, or `UPDATE` leading to data loss or modification.
🛡️ **Solution:** The query execution has been wrapped in a strictly read-only transaction (`SET TRANSACTION READ ONLY`) and a `rollback()` is explicitly executed after fetching rows, preventing any state modification regardless of the SQL string provided.
