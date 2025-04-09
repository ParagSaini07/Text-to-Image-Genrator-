document.getElementById("GenerateBtn").addEventListener("click", async function () {
    let token = ""
     let input = document.getElementById("textInput").value; // Get input from user
     let imageContainer = document.getElementById("imageContainer");
     let downloadBtn = document.getElementById("downloadBtn")
   
     // Function to generate image
     async function generateImage(data) {
         console.log(data)
       const response = await fetch(
         "https://api-inference.huggingface.co/models/ZB-Tech/Text-to-Image",
         {
           headers: {
             Authorization: `Bearer ${token}`,
           },
           method: "POST",
           body: JSON.stringify({ inputs: data }), // Pass data to inputs
         }
       );
       console.log(response)
   
       if (!response.ok) {
         throw new Error("API error");
       }
   
       const result = await response.blob(); // Get the image as a blob
       return result;
     }
   
     try {
       let imageBlob = await generateImage(input); // Pass 'input' to generateImage
       let imageUrl = URL.createObjectURL(imageBlob); // Create URL for the image blob
       imageContainer.innerHTML = `<img src="${imageUrl}" class="mx-auto rounded-lg shadow-lg" />`; // Display the image
 
       //add a download btn
       const downloadBtn = document.createElement('button');
       downloadBtn.textContent = "Download Image" //set the text of the btn
       downloadBtn.classList.add('w-full','bg-blue-500','rounded-lg','px-6',"mt-4",'py-2')
 
 
       downloadBtn.addEventListener("click",function(){
 
         const link = document.createElement('a')
         link.href = imageUrl;
         link.download = 'generated-image.png'
         link.click()
 
       })
 
       imageContainer.appendChild(downloadBtn);
 
 
     } catch (error) {
       console.error("Error generating image:", error);
     }
   });
   