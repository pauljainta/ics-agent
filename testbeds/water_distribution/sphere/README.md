# Water Distribution SPHERE Runtime

This directory intentionally contains only non-secret notes. Concrete SPHERE
deployment coordinates, XDC names, revision identifiers, node addresses, and
portal-specific values must live in the ignored local infrastructure profile:

```text
configs/private/sphere.local.yaml
```

Use `configs/infrastructure/sphere.example.yaml` as the template. Tracked
experiment configs should reference only the logical profile name
`water_distribution`.
