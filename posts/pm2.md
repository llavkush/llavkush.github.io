*How I went from juggling multiple terminals, crashed processes and manual server restarts to a single, calm workflow, using one lightweight tool.*

**5 minute read**

---

A few months ago I was working on several projects at once. Some were built with Node.js, some with Flask and FastAPI, and most had a frontend and a backend running as separate processes. The applications themselves were never the problem: *managing* them was.

My day kept turning into a loop of running things by hand:

```bash
node app.js
# or
python app.py
# or
npm run dev
```

Everything worked perfectly, until it didn't. And that's where **PM2** entered the picture.

---

## Situation

At first my workflow seemed fine: open a terminal, start the backend, open another terminal, start the frontend, SSH into a server, start a service manually, and repeat. It scaled badly the moment reality showed up.

An application would crash unexpectedly. A VM would reboot after maintenance. A process would consume too much memory and get killed by the OS. Or I'd SSH into a machine only to discover the service hadn't been running for hours.

The fix was always trivial (`node app.js` or `python app.py` again), but having to *notice* the failure and do it by hand never was. None of these were huge problems on their own. They were lots of small problems quietly stacking up, and the more services I ran, the heavier that pile got.

My first instinct was the classic trick: background the process with `nohup` so it survives the terminal closing:

```bash
nohup node app.js > app.log 2>&1 &
```

It works, but only just. `nohup` detaches the process and nothing more: it won't restart the app if it crashes, it won't bring it back after a reboot, it gives you no status, no resource usage, and no way to manage several services together. You're left grepping a log file and reading `ps` output to figure out whether anything is even alive. I needed something that actually *manages* processes, not just one that walks away from them.

---

## Task

I needed one tool that could:

- **Automatically restart** crashed applications
- **Centralise logs** so they live in one place
- **Monitor** CPU and memory usage
- **Survive reboots**: start services automatically after the server comes back
- **Manage many services** together as a group
- Work with **both Python and Node.js**
- Stay **lightweight and quick** to set up

Most importantly, I didn't want to add unnecessary complexity. For a lot of these projects, spinning up Kubernetes or building a full container-orchestration pipeline felt like overkill. I wanted something *practical*.

---

## Action

That's when I started using PM2. I'd assumed it was a Node.js-only tool, and I was completely wrong.

PM2 is, at its core, a **process manager**: if you can start an application with a command, PM2 can usually keep it alive, watch it and restart it. Installation took less than a minute:

```bash
npm install -g pm2
```

From there, things got interesting.

---

## Managing a Node.js application

Take a simple Express app that crashes occasionally on purpose:

```javascript
const express = require("express");
const app = express();

app.get("/", (req, res) => {
  res.send("Application Running");
});

// Simulate occasional crashes
setInterval(() => {
  if (Math.random() > 0.7) {
    console.log("Simulating crash...");
    process.exit(1);
  }
}, 30000);

app.listen(3000, () => {
  console.log("Server started on port 3000");
});
```

Normally I'd run `node app.js` and babysit it. With PM2 I start it once and name it:

```bash
pm2 start app.js --name node-api
```

Now when the process calls `process.exit(1)`, PM2 detects the failure and restarts it automatically: no SSH login, no manual restart, no surprise downtime.

---

## Monitoring applications

The first command I started using every day was:

```bash
pm2 list
```

It instantly shows process **status, CPU, memory, uptime and restart count** for everything PM2 manages. When you're running several services on one box, that single table becomes invaluable.

For a live view, PM2 ships a built-in dashboard:

```bash
pm2 monit
```

This gives a real-time picture of CPU utilisation, memory consumption, running services and overall health, genuinely useful for troubleshooting without reaching for a separate monitoring stack.

---

## Centralised logs

Before PM2 I'd `tail -f app.log` or dig through log directories per service. With PM2 every stream is in one place:

```bash
# all services
pm2 logs

# a single service
pm2 logs node-api
```

When you're debugging a production issue, having logs centralised saves a surprising amount of time.

---

## PM2 isn't just for Node.js

This was the biggest surprise for me: PM2 works just as well with Python. Point it at the right interpreter and you're done.

A Flask app:

```bash
pm2 start app.py --interpreter python3 --name flask-api
```

A FastAPI app started through its server command:

```bash
pm2 start "uvicorn main:app --host 0.0.0.0 --port 8000" \
  --interpreter bash \
  --name fastapi-api
```

You still get automatic restarts, monitoring, logs, startup recovery and group management, exactly the same experience as with Node.

---

## My favourite use case: frontend + backend together

This is where PM2 became part of my daily workflow. Most projects today have a frontend *and* a backend, which traditionally meant a window for React, one for Flask, one for FastAPI, and one for whatever else. PM2 lets me drive all of it from a single place.

```bash
# backend
pm2 start app.py --interpreter python3 --name backend

# frontend
pm2 start "npm run dev" --interpreter bash --name frontend
```

Then I manage them as one fleet:

```bash
pm2 list        # see everything at a glance
pm2 restart all # restart both
pm2 stop all    # stop both
pm2 delete all  # remove both
```

`pm2 list` prints something like:

```text
frontend    online
backend     online
```

That alone removed a lot of repetitive terminal-wrangling from my day.

---

## Using all CPU cores

By default a Node.js process runs on a single core. PM2 can fork multiple instances and load-balance across them with one flag:

```bash
pm2 start app.js -i max
```

This uses every available CPU core and distributes requests between the instances, no application changes required.

---

## Surviving server reboots

A lesson many of us learn only after deploying: **the server restarts, but the application doesn't.** PM2 closes that gap in two commands:

```bash
pm2 save     # snapshot the current process list
pm2 startup  # generate & install the boot script
```

After this, your saved services come back automatically whenever the machine boots. Simple, but the kind of thing that quietly prevents 2 a.m. incidents.

---

## Where PM2 fits compared to Docker

Whenever I mention PM2, someone asks the obvious question:

> Why not just use Docker?

They solve different problems. **Docker** packages an application and its dependencies into an isolated, portable container. **PM2** keeps a process alive and manageable on whatever host it's already on.

Docker is the right call when you need environment consistency, portability, containerised deployments and reproducibility across environments. But for many small and medium projects it adds overhead: building images, managing containers, networking, registries and extra resource usage.

Sometimes all you actually need is: start the app, keep it running, restart it if it crashes, watch its resources, and read its logs. For those cases PM2 is a much lighter answer:

```bash
pm2 start app.py --interpreter python3 --name backend
pm2 start "npm run dev" --interpreter bash --name frontend
```

No Dockerfile, no image builds, no orchestration, just running applications.

That said, PM2 doesn't *replace* Docker; in production they often complement each other:

```text
Docker → PM2 → Node.js application
Nginx  → PM2 → Node.js / Python application
```

The right tool depends on the problem you're solving.

---

## PM2 Plus (premium monitoring)

The free version is more than enough for most personal projects and plenty of production workloads. For larger setups, **PM2 Plus** adds centralised dashboards, real-time monitoring, alerts, advanced metrics, distributed visibility and team collaboration. If you're running many applications across many servers, it's worth a look.

- PM2 Runtime: <https://pm2.io/>
- PM2 Documentation: <https://pm2.keymetrics.io/>
- PM2 Plus: <https://pm2.io/plus/>

---

## The commands I reach for most

```bash
pm2 start app.js --name app   # start an app
pm2 list                      # list running apps
pm2 logs                      # view logs
pm2 monit                     # live resource monitor
pm2 restart app               # restart one app
pm2 restart all               # restart everything
pm2 stop all                  # stop everything
pm2 delete all                # remove everything
pm2 save                      # save the current process list
pm2 startup                   # start on boot
```

These cover roughly 90% of my day-to-day PM2 usage.

---

## Result

After switching to PM2:

- Applications recover from crashes on their own
- Monitoring and troubleshooting got easier
- Logs are centralised
- Better CPU utilisation
- Services start automatically after reboots
- Frontend/backend workflows are simpler
- Far less time spent managing terminals

Most importantly, I stopped worrying about whether my applications were even still running.

---

## What I learned

When developers talk about infrastructure, the conversation jumps straight to Docker, Kubernetes, service meshes and cloud-native platforms. Those tools absolutely have their place, but sometimes the highest-ROI improvement is solving a much smaller question first:

> "What happens when my application crashes?"

For me, PM2 answered that in under ten minutes, and as a bonus became my default way of launching and managing both backend and frontend services. Not every useful tool needs to be complex: some of the best ones quietly do their job and stay out of your way. PM2 is one of those.

---

**Do you use PM2, Docker, systemd, Supervisor, or something else to manage your applications? I'd love to hear what's working for you.**
