import express from 'express'
import pg from 'pg'
import { readFileSync } from 'node:fs'
import tls from 'node:tls'


// ---- PostgreSQL connection: credentials come from the service binding (VCAP_SERVICES) ----
// Locally there is no VCAP_SERVICES: read the same JSON from vcap.local.json and go through the SSH tunnel.
const local = !process.env.VCAP_SERVICES
const vcap = JSON.parse(local ? readFileSync('./vcap.local.json', 'utf8') : process.env.VCAP_SERVICES)
const creds = vcap['postgresql-db'][0].credentials

const pool = new pg.Pool({
  host: local ? 'localhost' : creds.hostname,
  port: local ? Number(process.env.PG_TUNNEL_PORT || 63306) : Number(creds.port),
  database: creds.dbname,
  user: creds.username,
  password: creds.password,
  ssl: {
    ca: creds.sslrootcert, // BTP PostgreSQL requires SSL
    // through the tunnel we connect to "localhost", but the certificate belongs to the real DB host
    checkServerIdentity: (_host, cert) => tls.checkServerIdentity(creds.hostname, cert)
  }
})
const GEMINI_KEY = process.env.GEMINI_API_KEY
const MODEL = 'gemini-embedding-001'
const DIMS = 768 // must match vector(768) below
const DOC_TASK   = process.env.DOC_TASK   || 'RETRIEVAL_DOCUMENT'
const QUERY_TASK = process.env.QUERY_TASK || 'RETRIEVAL_QUERY'
async function embed(texts, taskType) {
  const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:batchEmbedContents`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'x-goog-api-key': GEMINI_KEY },
    body: JSON.stringify({
      requests: texts.map(text => ({
        model: `models/${MODEL}`,
        content: { parts: [{ text }] },
        taskType,                    // RETRIEVAL_DOCUMENT for records, RETRIEVAL_QUERY for searches
        outputDimensionality: DIMS
      }))
    })
  })
  if (!res.ok) throw new Error(`Gemini ${res.status}: ${await res.text()}`)
  const out = await res.json()
  return out.embeddings.map(e => `[${e.values.join(',')}]`) // pgvector text format
}


const app = express()

// 1) Create table + load sample customers
app.post('/setup', async (req, res) => {
  try {
    await pool.query('CREATE EXTENSION IF NOT EXISTS vector')
  } catch (e) {
    return res.status(500).json({ step: 'enable vector extension', error: e.message,
      hint: 'Enable it via the BTP backing-services API, then call /setup again.' })
  }
  await pool.query(`CREATE TABLE IF NOT EXISTS customers (
    id           SERIAL PRIMARY KEY,
    company_name TEXT NOT NULL,
    description  TEXT NOT NULL,
    embedding    vector(${DIMS})
  )`)
  const { rows } = await pool.query('SELECT COUNT(*)::int AS n FROM customers')
  if (rows[0].n === 0) {
    const data = JSON.parse(readFileSync('./customers.json', 'utf8'))
    for (const c of data) {
      await pool.query('INSERT INTO customers (company_name, description) VALUES ($1, $2)', [c.company_name, c.description])
    }
  }
  res.json({ ok: true })
})

// 2) Vectorize every record that has no embedding yet and store the vector
app.post('/embed', async (req, res) => {
  const { rows } = await pool.query('SELECT id, company_name, description FROM customers WHERE embedding IS NULL')
  if (!rows.length) return res.json({ embedded: 0 })
  const vectors = await embed(rows.map(r => `${r.company_name}: ${r.description}`), DOC_TASK)
  for (let i = 0; i < rows.length; i++) {
    await pool.query('UPDATE customers SET embedding = $1::vector WHERE id = $2', [vectors[i], rows[i].id])
  }
  res.json({ embedded: rows.length })
})

// 3) Vectorize the query, compare with stored vectors (<=> = cosine distance), return closest records
app.get('/search', async (req, res) => {
  const q = req.query.q
  const top = Math.min(Number(req.query.top) || 3, 10)
  if (!q) return res.status(400).json({ error: 'use /search?q=your question' })

    const [qv] = await embed([q], QUERY_TASK)
  const { rows } = await pool.query(
    `SELECT id, company_name, description,
            ROUND((1 - (embedding <=> $1::vector))::numeric, 4) AS similarity
       FROM customers
      WHERE embedding IS NOT NULL
      ORDER BY embedding <=> $1::vector
      LIMIT $2`, [qv, top])
  res.json({ query: q,dimensions: DIMS ,MODEL, QUERY_TASK,DOC_TASK, results: rows,  })
})

app.get('/test', async (req, res) => {
  const texts = [
    'Recipe for banana bread with walnuts.',
    'The theory of general relativity describes gravity as curved spacetime.',
    'Freight forwarding and last-mile delivery for e-commerce shops.',
    'The football match ended two to one after extra time.'
  ]
  const vecs = (await embed(texts, 'RETRIEVAL_DOCUMENT')).map(v => JSON.parse(v))
  const cos = (a, b) => {
    const dot = a.reduce((s, x, i) => s + x * b[i], 0)
    return dot / (Math.hypot(...a) * Math.hypot(...b))
  }
  const pairs = []
  for (let i = 0; i < texts.length; i++)
    for (let j = i + 1; j < texts.length; j++)
      pairs.push({ a: texts[i].slice(0, 30), b: texts[j].slice(0, 30), similarity: cos(vecs[i], vecs[j]).toFixed(4) ,DIMS})
  res.json(pairs)
})
// Verification: compare stored customer vectors in JavaScript vs PostgreSQL 
app.get('/verify', async (req, res) => {
  // Read the same stored vectors that /search uses
  const { rows } = await pool.query(
    'SELECT id, company_name, embedding::text AS emb FROM customers WHERE embedding IS NOT NULL ORDER BY id')
  const vecs = rows.map(r => JSON.parse(r.emb))

  // Cosine similarity, computed by Js (same formula as pgvector's <=>)
  const cos = (a, b) => {
    const dot = a.reduce((s, x, i) => s + x * b[i], 0)
    return dot / (Math.hypot(...a) * Math.hypot(...b))
  }

  //PostgreSQL's own comparison for every pair
  const { rows: pgRows } = await pool.query(
    `SELECT a.id AS a, b.id AS b, 1 - (a.embedding <=> b.embedding) AS sim
       FROM customers a JOIN customers b ON a.id < b.id
      WHERE a.embedding IS NOT NULL AND b.embedding IS NOT NULL`)
  const pgSim = Object.fromEntries(pgRows.map(r => [`${r.a}-${r.b}`, Number(r.sim)]))

  // JavaScript vs PostgreSQL
  const pairs = []
  for (let i = 0; i < rows.length; i++)
    for (let j = i + 1; j < rows.length; j++) {
      const js = cos(vecs[i], vecs[j])
      const pg = pgSim[`${rows[i].id}-${rows[j].id}`]
      pairs.push({ a: rows[i].company_name, b: rows[j].company_name,
                   js: +js.toFixed(4), postgres: +pg.toFixed(4)})
    }

  res.json({
    pairs: pairs.sort((x, y) => y.js - x.js)
  })
})

// Just to look at the data (vectors left out)
app.get('/customers', async (req, res) => {
  const { rows } = await pool.query('SELECT id, company_name, description, embedding IS NOT NULL AS has_vector FROM customers ORDER BY id')
  res.json(rows)
})

app.use((err, req, res, next) => res.status(500).json({ error: err.message }))
app.listen(process.env.PORT || 8080, () => console.log('pg-vector app running'))