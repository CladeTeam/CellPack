# Training

These programs show how Python applications supply data to training. Source is
read-only in the documentation. Run it in your own prepared environment; these
examples are not submitted to the website's fixed Rust example service.

<DocDiagram name="training-start" />

Start with the DataLoader example for ordinary training iteration. Use the
lower-level integration when your application needs explicit native control.

Read [batch ownership and cleanup](/python/api/dataloader.md#iteration-epochs-and-completion)
before retaining outputs or configuring asynchronous transfers.
