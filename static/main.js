 let input = document.querySelector('div.app input')
 let img = document.querySelector('div.app img')
 let app = document.querySelector("div.app")

 let loaderDiv = document.querySelector('#c-images .loaderDiv')

 let resetBtn = document.querySelector('#reset')

let detectedImgDiv = document.querySelector('#c-images')

    input.addEventListener('paste',()=>{
        setTimeout(()=>{
          img.src = input.value
      },500)
    })

let xhr = new XMLHttpRequest()  // one for the detection

let xhrP = new XMLHttpRequest() // one for the prediction

  let imgsEncoded  = undefined

  let detectedImgCount = undefined


  resetBtn.addEventListener('click',()=>{
    let imgDiv = detectedImgDiv.querySelectorAll('div')

    for(let i = 0;i<imgDiv.length;i++){
        imgDiv[i].style.animation = 'scaledown 0.5s'
    }

    setTimeout(()=>{
        detectedImgDiv.innerHTML = ''
        detectedImgDiv.innerHTML = "<div class='loaderDiv'><div class='loader'></div></div>"
        detectedImgDiv.style.display = 'none'
          app.style.left = "30%"
          resetBtn.style.display = 'none'
      },400)

  })


  // we listen for a click event on the button
  button.addEventListener('click',()=>{
     app.style.left = '0'
     detectedImgDiv.style.display = 'flex'
     setTimeout(()=>{

        perform()
        resetBtn.style.display = 'block'
    }, 1200)

})


function perform(){
     let form = new FormData()

     form.append('imgURL',input.value)

     xhr.open('post','/detect')

     xhr.onload = function(){
         try{
             loaderDiv.style.display = 'none'
             imgsEncoded = JSON.parse(xhr.responseText)

             detectedImgCount = -1

            // Looping through the images to remove useless strings
             for(key in imgsEncoded){
                 imgsEncoded[key] = imgsEncoded[key].replaceAll("'","")
                 imgsEncoded[key] = imgsEncoded[key].replace("b","")
                 imgsEncoded[key]="data:image/jpeg;base64,"+imgsEncoded[key]
                 detectedImgCount+=1
             }

             img.src=imgsEncoded['0']
             // console.log(imgsEncoded)

             if(detectedImgCount > 3){
                detectedImgDiv.style.overflowY = "scroll"
             }else{
              detectedImgDiv.style.overflowY = ''
             }


            detectedImgDiv.innerHTML = ""
             for(key in imgsEncoded){
                 if(key == 0){
                    continue
                 }
                 else{
                       let width = (detectedImgCount - 1 > 1) ? 30 : 50
                       detectedImgDiv.innerHTML+= `<div id="cropped-img-div" style="width:${width}%">
                                                                                         <img width=${100}% src=${imgsEncoded[key]}>
                                                                                    </div>`
                 }
             }
         }
        catch{
             console.log("Could not  detect any frontal face")
        }


         if(imgsEncoded != undefined){

       //  THEN WE WILL SEND THE DETECTED IMAGES TO THE "PREDICT URL" for prediction.

             let form = new FormData()

             for(let i=0;i<detectedImgCount;i++){
                form.append(`${i+1}`, imgsEncoded[i+1])
             }

            xhrP.open('post','/predict')

            xhrP.onload = ()=>{
                let preds = JSON.parse(xhrP.responseText)

                let predsDiv = detectedImgDiv.querySelectorAll('div')

                for(let i=0;i<predsDiv.length;i++){
                    let span = document.createElement('span')
                    span.textContent = preds[i]
                    predsDiv[i].appendChild(span)
                }

            }

            xhrP.send(form)   // SENDING REQUESTS TO THE PREDICTING URL
      }
 }

   xhr.send(form)
}
