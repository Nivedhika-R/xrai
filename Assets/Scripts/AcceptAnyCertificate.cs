
using UnityEngine.Networking;

class AcceptAnyCertificate : CertificateHandler {
    protected override bool ValidateCertificate(byte[] certificateData) => true;
}
