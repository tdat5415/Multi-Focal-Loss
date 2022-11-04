# Multi-Focal-Loss


```python
class MultiFocalLoss(tf.keras.losses.Loss):
    def __init__(self, class_weights=None, gamma=2, eps=1e-7, name="MultiFocalLoss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_weights = class_weights
        self.gamma = gamma
        self.eps = eps
        
    def call(self, y_true, y_pred):
        pos_loss = y_true * tf.pow(1-y_pred, self.gamma) * tf.math.log(y_pred+self.eps)
        neg_loss = (1-y_true) * tf.pow(y_pred, self.gamma) * tf.math.log(1-y_pred+self.eps)
        loss = -(pos_loss + neg_loss) # (B, n_classes)
        if self.class_weights is None: loss = tf.reduce_sum(loss, axis=1)
        else: loss = tf.reduce_sum(loss*self.class_weights, axis=1)
        loss = tf.reduce_mean(loss, axis=0)
        return loss
```

---

##### example

```python
model.compile(loss=MultiFocalLoss(), optimizer='adam', metrics=['acc'])
```
or
```python
class_weights = [...]
loss = MultiFocalLoss(class_weights)
model.compile(loss=loss, optimizer='adam', metrics=['acc'])
```
