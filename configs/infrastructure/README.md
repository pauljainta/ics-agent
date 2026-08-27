# configs/infrastructure

This folder contains public infrastructure templates only.

## Files

- [`sphere.example.yaml`](sphere.example.yaml): placeholder SPHERE/OpenPLC runtime profile for water-treatment and water-distribution testbeds.

## Local Use

Copy the example into the ignored private config directory:

```bash
cp configs/infrastructure/sphere.example.yaml configs/private/sphere.local.yaml
```

Fill the copy with your active reservation details. Do not commit the filled-in file.

The runtime loads local infrastructure through `ICS_AGENT_SPHERE_CONFIG`, shown in [`.env.example`](../../.env.example):

```bash
ICS_AGENT_SPHERE_CONFIG=configs/private/sphere.local.yaml
```

## What Belongs Here

Tracked infrastructure docs and examples should contain placeholders, schema structure, and safe defaults. Real account names, reservations, XDC names, private hosts, and addresses do not belong here.
