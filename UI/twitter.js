const wrapper = document.querySelector(".wrapper"),
editableInput = wrapper.querySelector(".editable"),
readonlyInput = wrapper.querySelector(".readonly"),
placeholder = wrapper.querySelector(".placeholder"),
counter = wrapper.querySelector(".counter"),
button = wrapper.querySelector("button");
const xmark = document.querySelector(".fa-circle-xmark");
const check = document.querySelector(".fa-circle-check");
xmark.style.display = "none"
check.style.display = "none"
editableInput.onfocus = ()=>{
  placeholder.style.color = "#c5ccd3";
}
editableInput.onblur = ()=>{
  placeholder.style.color = "#98a5b1";
}
editableInput.onkeyup = (e)=>{
  let element = e.target;
  validated(element);
}
editableInput.onkeypress = (e)=>{
  let element = e.target;
  validated(element);
  placeholder.style.display = "none";
}
function validated(element){
  let text;
  let maxLength = 280;
  let currentlength = element.innerText.length;
  if(currentlength <= 0){
    placeholder.style.display = "block";
    counter.style.display = "none";
    button.classList.remove("active");
  }else{
    placeholder.style.display = "none";
    counter.style.display = "block";
    button.classList.add("active");
  }
  counter.innerText = maxLength - currentlength;
  if(currentlength > maxLength){
    let overText = element.innerText.substr(maxLength); 
    overText = `<span class="highlight">${overText}</span>`; 
    text = element.innerText.substr(0, maxLength) + overText; 
    readonlyInput.style.zIndex = "1";
    counter.style.color = "#e0245e";
    button.classList.remove("active");
  }else{
    readonlyInput.style.zIndex = "-1";
    counter.style.color = "#333";
  }
  readonlyInput.innerHTML = text; 

  
const submitBtn = document.getElementById("submit-btn");
submitBtn.addEventListener("click", function() {
    const tweet = document.querySelector(".editable").innerText;
    
    fetch("http://127.0.0.1:8000/predict", {
        method: "post",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ "tweet": tweet })
    })
    .then(res => res.json())
    .then(data => {
        if (data.prediction === 0) {
          check.style.display = "none";
          xmark.style.display = "block";
          result.innerHTML = "I am " + data.probability*100 + "% sure about it" ;
;
        } else {
          check.style.display = "block";
          xmark.style.display = "none";
          result.innerHTML = "I am " + data.probability*100 + "% sure about it" ;
;
        }
    })
    .catch(err => {
        console.error(err);
        result.innerHTML = "Error making request to the API";
    });
});
  
  
}
