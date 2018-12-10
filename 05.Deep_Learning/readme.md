## Deep Learning
---

**How to Save a Model ?**
---

1. *Method 1:*
    ```python
    model.save("modelName.hdf5")
    ```
1. *Method 2:*
    ```python
    from keras.models import model_from_json
    from keras.models import load_model
    
    model_json = model.to_json()
        
    with open("model_json.json", "w") as json_file:
        json_file.write(model_json)
     
    model.save_weights("model_num.h5")
    ```

**How to Save a Model ?**
---

1. *Method 1:*
    ```python
    load_model("modelName.hdf5")
    ```
1. *Method 2:*
    ```python
    
    ```    