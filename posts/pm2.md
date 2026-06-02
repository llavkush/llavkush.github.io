*How I went from managing multiple terminals, crashed processes, and server restarts to a much simpler workflow.*

**3-5 minute read**

---

A few months ago, I was working on multiple projects at the same time.

Some were built with Node.js.

Some were built with Flask and FastAPI.

Most had a frontend and backend running separately.

The applications themselves were not the problem.

Managing them was.

I kept finding myself doing this:

```bash
node app.js
```

or

```bash
python app.py
```

or

```bash
npm run dev
```

Everything worked perfectly...

Until it didn't.

And that's where PM2 entered the picture.

---

## Situation

At first, my workflow seemed fine.

Open a terminal.

Start the backend.

Open another terminal.

Start the frontend.

SSH into a server.

Start a service manually.

Repeat.

Then reality happened.

An application would crash unexpectedly.

A VM would reboot after maintenance.

A process would consume too much memory and get killed.

Or I'd SSH into a machine only to realize the application wasn't running anymore.

The fix was always simple:

```bash
node app.js
```

or

```bash
python app.py
```

But having to do it manually every time wasn't.

The more applications I managed, the more annoying it became.

Not a huge problem.

Just lots of small problems adding up.

---

## Task

I needed something that could:

* Automatically restart crashed applications
* Show me logs in one place
* Monitor CPU and memory usage
* Start applications automatically after server reboots
* Manage multiple services together
* Work with both Python and Node.js
* Be lightweight and easy to set up

Most importantly:

I didn't want to introduce unnecessary complexity.

For many of my projects, spinning up Kubernetes clusters or building full container orchestration pipelines felt like overkill.

I wanted something practical.

---

## Action

That's when I started using PM2.

Initially, I thought PM2 was only for Node.js developers.

Turns out I was completely wrong.

PM2 is essentially a process manager.

If you can start an application using a command, PM2 can usually manage it.

Installation took less than a minute:

```bash
npm install -g pm2
```

And from there, things got interesting.

---

## Managing a Node.js Application

Let's take a simple Express application.

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

Normally, I would run:

```bash
node app.js
```

With PM2:

```bash
pm2 start app.js --name node-api
```

Now if the application crashes:

```javascript
process.exit(1);
```

PM2 automatically detects the failure and restarts it.

No SSH login.

No manual restart.

No surprise downtime.

---

## Monitoring Applications

One of the first commands I started using daily:

```bash
pm2 list
```

This immediately shows:

* Process status
* CPU usage
* Memory usage
* Uptime
* Restart count

When you're running multiple applications on a server, this becomes incredibly useful.

---

## Real-Time Monitoring

PM2 also provides a built-in dashboard.

```bash
pm2 monit
```

This gives a real-time view of:

* CPU utilization
* Memory consumption
* Running services
* Application health

It's surprisingly useful for troubleshooting without needing additional monitoring tools.

---

## Centralized Logs

Before PM2, I often found myself doing things like:

```bash
tail -f app.log
```

or searching through log directories.

With PM2:

```bash
pm2 logs
```

Or for a specific service:

```bash
pm2 logs node-api
```

Everything is centralized.

When debugging production issues, this saves a surprising amount of time.

---

## PM2 Isn't Just for Node.js

This was probably the biggest surprise for me.

PM2 works extremely well with Python applications too.

A Flask application:

```bash
pm2 start app.py \
--interpreter python3 \
--name flask-api
```

A FastAPI application:

```bash
pm2 start "uvicorn main:app --host 0.0.0.0 --port 8000" \
--interpreter bash \
--name fastapi-api
```

You still get:

* Automatic restarts
* Monitoring
* Logs
* Startup recovery
* Process management

Exactly the same experience.

---

## My Favorite Use Case: Frontend + Backend Together

This is where PM2 became part of my daily workflow.

Most projects today have:

* Frontend
* Backend

Traditionally, I would have multiple terminal windows open.

One for React.

One for Flask.

One for FastAPI.

One for something else.

Instead, PM2 lets me manage everything from a single place.

Start backend:

```bash
pm2 start app.py \
--interpreter python3 \
--name backend
```

Start frontend:

```bash
pm2 start "npm run dev" \
--interpreter bash \
--name frontend
```

Check everything:

```bash
pm2 list
```

Example:

```text
frontend    online
backend     online
```

Restart both:

```bash
pm2 restart all
```

Stop both:

```bash
pm2 stop all
```

Delete both:

```bash
pm2 delete all
```

This alone eliminated a lot of repetitive work for me.

---

## Using All CPU Cores

By default, Node.js runs on a single CPU core.

PM2 can automatically launch multiple instances.

```bash
pm2 start app.js -i max
```

This uses all available CPU cores and load-balances requests between instances.

A single command.

No application changes required.

---

## Surviving Server Reboots

One lesson many developers learn only after deploying:

The server restarts.

The application doesn't.

PM2 solves this with:

```bash
pm2 save
pm2 startup
```

Now applications automatically start whenever the machine boots.

Simple.

But incredibly useful.

---

## Where PM2 Fits Compared to Docker

Whenever I mention PM2, someone inevitably asks:

> Why not just use Docker?

The reality is that PM2 and Docker solve different problems.

Docker focuses on packaging applications and their dependencies into isolated containers.

PM2 focuses on keeping processes alive and manageable.

Docker is fantastic when you need:

* Environment consistency
* Portability
* Containerized deployments
* Multi-environment reproducibility

But for many small and medium-sized projects, Docker can introduce additional overhead:

* Building images
* Managing containers
* Networking configuration
* Container registries
* Additional storage and resource usage

Sometimes all you really need is:

* Start application
* Keep it running
* Restart if it crashes
* Monitor resources
* View logs

For those situations, PM2 can be a very lightweight alternative.

For example:

```bash
pm2 start app.py --interpreter python3 --name backend

pm2 start "npm run dev" --interpreter bash --name frontend
```

And you're done.

No Dockerfile.

No image builds.

No container orchestration.

Just working applications.

That doesn't mean PM2 replaces Docker.

In many production environments they complement each other perfectly.

Examples:

```text
Docker → PM2 → Node.js Application
```

or

```text
Nginx → PM2 → Node.js/Python Application
```

The right tool depends on the problem you're solving.

---

## PM2 Plus (Premium Monitoring)

The free version of PM2 is more than enough for most personal projects and many production workloads.

However, PM2 also offers a premium monitoring platform called PM2 Plus.

Features include:

* Centralized dashboards
* Real-time monitoring
* Alerts
* Advanced metrics
* Distributed application visibility
* Team collaboration

If you're managing multiple applications across multiple servers, it's worth exploring.

Useful links:

* PM2 Runtime: https://pm2.io/
* PM2 Documentation: https://pm2.keymetrics.io/
* PM2 Plus: https://pm2.io/plus/

---

## PM2 Commands I Use Most Often

Start application:

```bash
pm2 start app.js --name app
```

List running applications:

```bash
pm2 list
```

View logs:

```bash
pm2 logs
```

Monitor resources:

```bash
pm2 monit
```

Restart application:

```bash
pm2 restart app
```

Restart everything:

```bash
pm2 restart all
```

Stop everything:

```bash
pm2 stop all
```

Delete everything:

```bash
pm2 delete all
```

Save configuration:

```bash
pm2 save
```

Enable startup on reboot:

```bash
pm2 startup
```

These commands probably cover 90% of my PM2 usage.

---

## Result

After switching to PM2:

✅ Applications automatically recover from crashes

✅ Easier monitoring and troubleshooting

✅ Centralized logs

✅ Better CPU utilization

✅ Automatic startup after server reboots

✅ Simpler frontend/backend workflows

✅ Less time managing terminals

Most importantly:

I stopped worrying about whether my applications were still running.

---

## What I Learned

When developers discuss infrastructure, the conversation often jumps directly to:

* Docker
* Kubernetes
* Service Meshes
* Cloud-Native Platforms

Those tools absolutely have their place.

But sometimes the highest ROI improvement comes from solving a much simpler problem:

> "What happens when my application crashes?"

For me, PM2 solved that problem in less than ten minutes.

And as a bonus, it became my default way of launching and managing both backend and frontend services.

Not every useful tool needs to be complex.

Some of the best ones quietly do their job and stay out of your way.

PM2 is one of those tools.

---

**Do you use PM2, Docker, systemd, Supervisor, or something else to manage your applications? I'd love to hear what's working for you.**

#NodeJS #PM2 #Python #Flask #FastAPI #ReactJS #DevOps #BackendDevelopment #SoftwareEngineering #Linux #WebDevelopment #CloudComputing
