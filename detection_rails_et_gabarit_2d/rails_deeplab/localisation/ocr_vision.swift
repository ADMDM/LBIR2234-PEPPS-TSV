import Foundation
import Vision
import ImageIO
for path in CommandLine.arguments.dropFirst() {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    request.recognitionLanguages = ["en-US"]
    request.usesCPUOnly = true
    do {
        let handler = VNImageRequestHandler(url: URL(fileURLWithPath:path), options: [:])
        try handler.perform([request])
        let lines = (request.results ?? []).compactMap { obs -> [String:Any]? in
            guard let c = obs.topCandidates(1).first else { return nil }
            return ["text":c.string,"confidence":c.confidence,"y":obs.boundingBox.minY]
        }
        let obj:[String:Any] = ["path":path,"lines":lines]
        let d=try JSONSerialization.data(withJSONObject:obj,options:[.sortedKeys])
        print(String(data:d,encoding:.utf8)!)
    } catch {
        let d = try! JSONSerialization.data(withJSONObject: ["path":path, "error":String(describing:error)], options:[.sortedKeys])
        print(String(data:d,encoding:.utf8)!)
    }
}
